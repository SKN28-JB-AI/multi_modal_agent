"""
pipeline/orchestrator.py
------------------------
잡 실행 오케스트레이터. 모드별 파이프라인을 단계대로 진행하며
JobManager 를 통해 상태를 갱신한다.

  메시지 모드: 프롬프트 1개 → 단일 씬 스토리보드 → ③ 생성 → ④ 후처리
  PDF 모드   : ① 파싱/이해 → ② 스토리보드 → ③ 씬별 생성 → ④ 후처리

[비용/안정성 가드]
- 씬 생성은 max_concurrent_clips 세마포어로 동시성 제한.
- 클립별 clip_retries 회 재시도. 한 씬이 끝내 실패하면 잡 전체 실패
  (부분 결합은 광고 결과물로 무의미하므로).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from ..backends import ClipSpec, get_backend
from ..config import Settings
from ..jobs import Job, JobManager, JobStatus, SceneState
from ..schemas import PdfJobOptions, Scene, Storyboard
from . import postprocess
from .pdf_parser import extract_pdf

logger = logging.getLogger(__name__)

# 보이스오버 지시문에 쓸 언어 표기 (비디오 모델은 영어 지시문 + 대상 언어명 조합이 가장 안정적)
_LANGUAGE_NAMES = {
    "ko": "Korean", "en": "English", "ja": "Japanese",
    "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German",
}


def get_llm(settings: Settings):
    """LLM 인스턴스 팩토리. 테스트에서 이 함수를 교체(mock)한다."""
    from ..llm import OpenAILLM

    return OpenAILLM(settings)


class Orchestrator:
    def __init__(self, settings: Settings, manager: JobManager) -> None:
        self.settings = settings
        self.manager = manager

    # ================================================================== #
    # 메시지 모드
    # ================================================================== #
    async def run_message_job(self, job: Job) -> None:
        req = job.request
        duration = req.get("duration_sec") or 6.0
        # 비디오 생성 전 프롬프트 변환(선행 단계). 실패해도 원본으로 진행.
        prompt = await self._maybe_enhance_prompt(job, duration)
        storyboard = Storyboard(
            title="message",
            scenes=[
                Scene(
                    index=0,
                    prompt=prompt,
                    duration_sec=duration,
                )
            ],
        )
        self.manager.update(job, storyboard=storyboard)
        await self._generate_and_finish(job, storyboard, options=None)

    async def _maybe_enhance_prompt(self, job: Job, duration: float) -> str:
        """
        메시지 모드 프롬프트를 비디오 모델 맞춤형으로 변환한다(선행 단계).

        OpenAI 기본 설정 모델(settings.openai_llm_model)을 사용한다.
        다음 경우에는 원본 프롬프트를 그대로 사용한다(안전 폴백):
          - 요청/설정에서 변환 비활성화
          - OPENAI_API_KEY 미설정
          - 변환 호출 실패(네트워크/쿼터 등)
        변환에 사용된 원본/결과는 job.request 에 기록해 감사 가능하게 한다.
        """
        req = job.request
        original = req["prompt"]

        # 요청값(enhance_prompt) 우선, 없으면 서버 기본값.
        want = req.get("enhance_prompt")
        if want is None:
            want = self.settings.enhance_message_prompt
        if not want:
            return original

        if not self.settings.openai_api_key:
            logger.info(
                "잡 %s: OPENAI_API_KEY 미설정 → 프롬프트 변환 생략(원본 사용)", job.id
            )
            return original

        try:
            llm = get_llm(self.settings)
            enhanced = await llm.enhance_video_prompt(
                original,
                model=job.model,
                aspect_ratio=req.get("aspect_ratio") or "16:9",
                resolution=req.get("resolution") or "1080p",
                duration_sec=duration,
                language="ko",
            )
        except Exception as exc:  # noqa: BLE001 - 변환 실패는 비치명(원본 폴백)
            logger.warning(
                "잡 %s: 프롬프트 변환 실패 → 원본 사용: %s", job.id, exc
            )
            return original

        enhanced = (enhanced or "").strip()
        if not enhanced or enhanced == original.strip():
            return original

        # 감사용 기록(원본 프롬프트 보존 + 변환 결과).
        req["original_prompt"] = original
        req["enhanced_prompt"] = enhanced
        self.manager.update(job, request=req)
        logger.info("잡 %s: 프롬프트 변환 완료(모델=%s)", job.id, job.model)
        return enhanced

    # ================================================================== #
    # PDF 기획서 모드
    # ================================================================== #
    async def run_pdf_job(self, job: Job, pdf_path: Path,
                          options: PdfJobOptions,
                          logo_path: Optional[Path] = None) -> None:
        # --- ① 파싱/이해 ---------------------------------------------- #
        self.manager.update(job, status=JobStatus.PARSING, progress=0.05)
        parsed = await asyncio.to_thread(
            extract_pdf, pdf_path,
            self.settings.pdf_max_pages, self.settings.pdf_render_dpi,
        )
        llm = get_llm(self.settings)
        brief = await llm.analyze_pdf(parsed.text, parsed.page_images)
        logger.info("잡 %s: 브리프 생성 완료", job.id)

        # --- ② 스토리보드 --------------------------------------------- #
        self.manager.update(job, status=JobStatus.STORYBOARDING, progress=0.15)
        storyboard = await llm.make_storyboard(brief, options)
        # 씬 index 를 순서대로 강제(LLM 출력 방어)
        for i, scene in enumerate(storyboard.scenes):
            scene.index = i
        self.manager.update(job, storyboard=storyboard)

        # --- ③ + ④ ---------------------------------------------------- #
        if options.generation_mode == "single":
            await self._generate_single_and_finish(
                job, storyboard, options, logo_path
            )
        else:
            await self._generate_and_finish(job, storyboard, options, logo_path)

    # ================================================================== #
    # remix 모드: 완료된 잡의 특정 씬을 부분 수정 후 재결합
    # ================================================================== #
    async def run_remix_job(self, job: Job, source_job: Job,
                            scene_index: int, prompt: str) -> None:
        """
        source_job 의 scene_index 클립을 백엔드 remix 로 교체하고,
        나머지 클립은 원본을 복사해 다시 결합한다.
        원본 잡 디렉터리의 narration.mp3 / logo.png 가 있으면 재적용한다.
        """
        import shutil as _shutil

        settings = self.settings
        backend = get_backend(job.model, settings)
        job_dir = self.manager.job_dir(job.id)
        clips_dir = job_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        storyboard = source_job.storyboard.model_copy(deep=True)
        scenes = sorted(storyboard.scenes, key=lambda s: s.index)
        # 클립 단위는 원본 잡의 SceneState 기준으로 처리한다.
        # (single 모드 잡은 스토리보드 씬이 여러 개여도 클립은 1개)
        states_src = sorted(source_job.scenes, key=lambda s: s.index)
        src_states = {s.index: s for s in states_src}
        target_state = src_states[scene_index]

        scene_states = [
            SceneState(index=st.index, status="completed",
                       backend_job_id=st.backend_job_id)
            for st in states_src
        ]
        self.manager.update(
            job, status=JobStatus.GENERATING, progress=0.2,
            storyboard=storyboard, scenes=scene_states,
        )

        # --- 1) 원본 클립 복사 ----------------------------------------- #
        clip_paths: dict[int, Path] = {}
        for st in states_src:
            src_clip = st.clip_path
            if not src_clip or not Path(src_clip).exists():
                self.manager.update(
                    job, status=JobStatus.FAILED,
                    error=f"원본 씬 {st.index} 클립 파일이 없습니다.",
                )
                return
            dst = clips_dir / f"scene_{st.index:02d}.mp4"
            await asyncio.to_thread(_shutil.copyfile, src_clip, dst)
            clip_paths[st.index] = dst

        # --- 2) 대상 씬 remix ------------------------------------------ #
        target_scene_state = next(
            st for st in scene_states if st.index == scene_index
        )
        target_scene_state.status = "generating"
        self.manager.persist(job)
        try:
            result = await backend.remix_clip(
                target_state.backend_job_id, prompt,
                clip_paths[scene_index],
            )
        except Exception as exc:  # noqa: BLE001
            target_scene_state.status = "failed"
            target_scene_state.error = str(exc)
            self.manager.update(
                job, status=JobStatus.FAILED,
                error=f"씬 {scene_index} remix 실패: {exc}",
                scenes=scene_states,
            )
            return
        target_scene_state.status = "completed"
        target_scene_state.clip_path = str(result.path)
        target_scene_state.backend_job_id = result.meta.get("backend_job_id")

        # 스토리보드에 수정 프롬프트 기록(이력 추적)
        for s in scenes:
            if s.index == scene_index:
                s.prompt = prompt

        # --- 3) 후처리: 재결합 + SRT + (있으면) 내레이션/로고 재적용 ---- #
        self.manager.update(
            job, status=JobStatus.POSTPROCESSING, progress=0.9,
            storyboard=storyboard, scenes=scene_states,
        )
        ordered_clips = [clip_paths[st.index] for st in states_src]
        # SRT 타이밍: 원본 잡이 저장한 씬 길이를 우선 사용
        if (source_job.scene_durations
                and len(source_job.scene_durations) == len(scenes)):
            durations = list(source_job.scene_durations)
        else:
            durations = [
                backend.normalize_duration(s.duration_sec) for s in scenes
            ]
        if result.duration_sec:
            if len(states_src) == 1:
                # 단일 클립 잡: 전체 타임라인을 새 길이에 맞춰 비례 조정
                total = sum(durations) or 1.0
                factor = result.duration_sec / total
                durations = [d * factor for d in durations]
            elif scene_index < len(durations):
                durations[scene_index] = result.duration_sec

        current = job_dir / "merged.mp4"
        await asyncio.to_thread(postprocess.concat_clips, ordered_clips, current)

        srt_path = job_dir / "subtitles.srt"
        has_srt = await asyncio.to_thread(
            postprocess.write_srt, storyboard, durations, srt_path
        )

        source_dir = self.manager.job_dir(source_job.id)
        narration = source_dir / "narration.mp3"
        if narration.exists():
            mixed = job_dir / "merged_narrated.mp4"
            await asyncio.to_thread(
                postprocess.mix_narration, current, narration, mixed
            )
            current = mixed
        logo = source_dir / "logo.png"
        if logo.exists():
            branded = job_dir / "merged_branded.mp4"
            await asyncio.to_thread(
                postprocess.overlay_logo, current, logo, branded,
                settings.logo_scale_ratio, settings.logo_opacity,
                settings.logo_position, settings.logo_margin_ratio,
                settings.logo_fade_in_sec,
            )
            current = branded

        final_path = job_dir / "final.mp4"
        if current != final_path:
            await asyncio.to_thread(_replace, current, final_path)

        self.manager.update(
            job, status=JobStatus.COMPLETED, progress=1.0,
            final_path=str(final_path),
            subtitles_path=str(srt_path) if has_srt else None,
            scenes=scene_states, scene_durations=durations,
        )
        logger.info("remix 잡 %s 완료: %s", job.id, final_path)

    # ================================================================== #
    # 공통: ③ 씬별 생성 → ④ 후처리
    # ================================================================== #
    async def _generate_and_finish(
        self, job: Job, storyboard: Storyboard,
        options: Optional[PdfJobOptions],
        logo_path: Optional[Path] = None,
    ) -> None:
        settings = self.settings
        req = job.request
        backend = get_backend(job.model, settings)
        job_dir = self.manager.job_dir(job.id)
        clips_dir = job_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        scenes = storyboard.scenes
        scene_states = [SceneState(index=s.index) for s in scenes]
        self.manager.update(
            job, status=JobStatus.GENERATING, scenes=scene_states, progress=0.2
        )

        aspect = (options.aspect_ratio if options else req.get("aspect_ratio")) or "16:9"
        resolution = (options.resolution if options else req.get("resolution")) or "1080p"
        gen_audio = (
            options.generate_audio if options
            else req.get("generate_audio", True)
        )
        language = options.language if options else "ko"
        # TTS 내레이션(enable_narration)을 쓸 때는 모델 발화를 빼서
        # 목소리가 이중으로 겹치지 않게 한다.
        embed_narration = not (options and options.enable_narration)

        semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_clips))
        results: dict[int, Path] = {}
        durations: dict[int, float] = {}
        done_count = 0
        progress_lock = asyncio.Lock()

        async def _one_scene(scene: Scene, state: SceneState) -> None:
            nonlocal done_count
            spec = ClipSpec(
                prompt=self._compose_prompt(scene, language, embed_narration),
                duration_sec=scene.duration_sec,
                aspect_ratio=aspect,
                resolution=resolution,
                generate_audio=gen_audio,
                index=scene.index,
            )
            out_path = clips_dir / f"scene_{scene.index:02d}.mp4"
            attempts = settings.clip_retries + 1
            async with semaphore:
                state.status = "generating"
                self.manager.persist(job)
                last_exc: Exception | None = None
                for attempt in range(attempts):
                    try:
                        result = await backend.generate_clip(spec, out_path)
                        results[scene.index] = result.path
                        durations[scene.index] = result.duration_sec
                        state.status = "completed"
                        state.clip_path = str(result.path)
                        state.backend_job_id = result.meta.get("backend_job_id")
                        break
                    except Exception as exc:  # noqa: BLE001 - 재시도 대상
                        last_exc = exc
                        logger.warning(
                            "잡 %s 씬 %d 시도 %d/%d 실패: %s",
                            job.id, scene.index, attempt + 1, attempts, exc,
                        )
                else:
                    state.status = "failed"
                    state.error = str(last_exc)
                    raise RuntimeError(
                        f"씬 {scene.index} 생성 실패: {last_exc}"
                    ) from last_exc
            async with progress_lock:
                done_count += 1
                # 생성 단계는 전체 진행률의 0.2 ~ 0.9 구간을 차지
                self.manager.update(
                    job, progress=0.2 + 0.7 * (done_count / len(scenes))
                )

        try:
            await asyncio.gather(
                *(_one_scene(s, st) for s, st in zip(scenes, scene_states))
            )
        except Exception as exc:
            self.manager.update(
                job, status=JobStatus.FAILED, error=str(exc), scenes=scene_states
            )
            return

        # --- ④ 후처리 (공통) -------------------------------------------- #
        ordered_clips = [results[s.index] for s in scenes]
        ordered_durations = [durations[s.index] for s in scenes]
        await self._finalize(
            job, storyboard, ordered_clips, ordered_durations,
            options, logo_path, scene_states,
        )

    # ================================================================== #
    # 단일 생성 모드 (generation_mode="single")
    # ================================================================== #
    async def _generate_single_and_finish(
        self, job: Job, storyboard: Storyboard,
        options: PdfJobOptions, logo_path: Optional[Path] = None,
    ) -> None:
        """
        스토리보드 전체를 샷 타임라인 프롬프트 하나로 합성해
        단 한 번의 생성 요청으로 영상을 만든다 (클립 결합 없음).

        - 총 길이는 백엔드 지원 최대치로 자동 보정된다.
        - SRT 타이밍은 실제 생성 길이에 맞춰 씬 길이를 비례 축소해 동기화.
        """
        settings = self.settings
        backend = get_backend(job.model, settings)
        job_dir = self.manager.job_dir(job.id)
        clips_dir = job_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        embed_narration = not options.enable_narration
        planned_total = sum(s.duration_sec for s in storyboard.scenes) or 1.0
        prompt = self._build_single_prompt(
            storyboard, options.language, embed_narration, planned_total
        )

        state = SceneState(index=0)
        self.manager.update(
            job, status=JobStatus.GENERATING, scenes=[state], progress=0.2
        )

        spec = ClipSpec(
            prompt=prompt,
            duration_sec=planned_total,   # 백엔드가 지원 값으로 보정
            aspect_ratio=options.aspect_ratio,
            resolution=options.resolution,
            generate_audio=options.generate_audio,
            index=0,
        )
        out_path = clips_dir / "scene_00.mp4"
        state.status = "generating"
        self.manager.persist(job)

        result = None
        last_exc: Exception | None = None
        for attempt in range(settings.clip_retries + 1):
            try:
                result = await backend.generate_clip(spec, out_path)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "잡 %s 단일 생성 시도 %d 실패: %s", job.id, attempt + 1, exc
                )
        if result is None:
            state.status = "failed"
            state.error = str(last_exc)
            self.manager.update(
                job, status=JobStatus.FAILED,
                error=f"단일 생성 실패: {last_exc}", scenes=[state],
            )
            return

        state.status = "completed"
        state.clip_path = str(result.path)
        state.backend_job_id = result.meta.get("backend_job_id")
        self.manager.update(job, progress=0.85, scenes=[state])

        # 실제 생성 길이에 맞춰 씬 타임라인을 비례 축소(SRT 동기화)
        actual = result.duration_sec or planned_total
        scale = actual / planned_total
        srt_durations = [s.duration_sec * scale for s in storyboard.scenes]

        await self._finalize(
            job, storyboard, [result.path], srt_durations,
            options, logo_path, [state],
        )

    @staticmethod
    def _build_single_prompt(
        storyboard: Storyboard, language: str,
        embed_narration: bool, planned_total: float,
    ) -> str:
        """
        스토리보드를 '샷 타임라인' 단일 프롬프트로 합성한다.

        [프롬프팅 규약]
        - 샷별 시간 구간을 명시해 모델이 장면 전환 타이밍을 따르게 한다.
        - 보이스오버는 따옴표 대사 + 언어 + 화자 지정(3요소)을 한 블록으로,
          샷 순서대로 나열한다.
        """
        lang_name = _LANGUAGE_NAMES.get(language, language)
        n = len(storyboard.scenes)
        lines = [
            f"A single continuous {int(round(planned_total))}-second "
            f"commercial video, told as {n} sequential shots with smooth, "
            f"seamless transitions:"
        ]
        cursor = 0.0
        for i, s in enumerate(storyboard.scenes, start=1):
            seg = (
                f"Shot {i} ({int(cursor)}-{int(cursor + s.duration_sec)}s): "
                f"{s.prompt.strip()}"
            )
            if s.audio_description.strip():
                seg += f" Audio: {s.audio_description.strip()}"
            lines.append(seg)
            cursor += s.duration_sec

        narrations = [
            s.narration.strip() for s in storyboard.scenes if s.narration.strip()
        ]
        if embed_narration and narrations:
            spoken = " then ".join(f'"{t}"' for t in narrations)
            lines.append(
                f"Voiceover: a warm, calm narrator speaks in {lang_name} "
                f"across the shots, in order: {spoken} "
                f"(clear voiceover speech, not on-screen text)"
            )
        return "\n".join(lines)

    # ================================================================== #
    # ④ 공통 후처리: 결합/SRT/번인/내레이션/로고 → final.mp4
    # ================================================================== #
    async def _finalize(
        self, job: Job, storyboard: Storyboard,
        ordered_clips: list[Path], srt_durations: list[float],
        options: Optional[PdfJobOptions], logo_path: Optional[Path],
        scene_states: list[SceneState],
    ) -> None:
        settings = self.settings
        job_dir = self.manager.job_dir(job.id)
        self.manager.update(
            job, status=JobStatus.POSTPROCESSING, progress=0.9,
            scenes=scene_states, scene_durations=srt_durations,
        )

        current = job_dir / "merged.mp4"
        await asyncio.to_thread(postprocess.concat_clips, ordered_clips, current)

        # SRT (씬 카피가 있을 때만)
        srt_path = job_dir / "subtitles.srt"
        has_srt = await asyncio.to_thread(
            postprocess.write_srt, storyboard, srt_durations, srt_path
        )

        # 자막 번인(옵션)
        if has_srt and options and options.burn_subtitles:
            burned = job_dir / "merged_subtitled.mp4"
            await asyncio.to_thread(
                postprocess.burn_subtitles, current, srt_path, burned
            )
            current = burned

        # 내레이션(옵션)
        if options and options.enable_narration and storyboard.narration_script:
            if not self.settings.openai_api_key:
                logger.warning("잡 %s: OPENAI_API_KEY 없음 — 내레이션 생략", job.id)
            else:
                narration = job_dir / "narration.mp3"
                await asyncio.to_thread(
                    postprocess.synthesize_narration,
                    storyboard.narration_script,
                    self.settings.openai_api_key,
                    self.settings.tts_model, self.settings.tts_voice,
                    narration,
                )
                mixed = job_dir / "merged_narrated.mp4"
                await asyncio.to_thread(
                    postprocess.mix_narration, current, narration, mixed
                )
                current = mixed

        # 로고 오버레이(옵션)
        if logo_path is not None and logo_path.exists():
            branded = job_dir / "merged_branded.mp4"
            await asyncio.to_thread(
                postprocess.overlay_logo, current, logo_path, branded,
                settings.logo_scale_ratio, settings.logo_opacity,
                settings.logo_position, settings.logo_margin_ratio,
                settings.logo_fade_in_sec,
            )
            current = branded

        final_path = job_dir / "final.mp4"
        if current != final_path:
            await asyncio.to_thread(_replace, current, final_path)

        self.manager.update(
            job,
            status=JobStatus.COMPLETED,
            progress=1.0,
            final_path=str(final_path),
            subtitles_path=str(srt_path) if has_srt else None,
            scenes=scene_states,
        )
        logger.info("잡 %s 완료: %s", job.id, final_path)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_prompt(scene: Scene, language: str = "ko",
                        embed_narration: bool = True) -> str:
        """
        씬 프롬프트 + 오디오 묘사 + 보이스오버 지시문을 합성한다.

        [내레이션 프롬프팅 규약]
        비디오 모델이 음성을 발화하게 하려면 ① 따옴표 안의 대사
        ② 언어 명시 ③ 화자 지정이 모두 필요하다. scene.narration 이
        있으면 이 형식의 보이스오버 지시문을 자동 부착한다.
        (enable_narration=True 로 TTS 를 합성할 때는 중복 발화를 막기
        위해 부착하지 않는다.)
        """
        prompt = scene.prompt.strip()
        if scene.audio_description.strip():
            prompt += f"\nAudio: {scene.audio_description.strip()}"
        if embed_narration and scene.narration.strip():
            lang_name = _LANGUAGE_NAMES.get(language, language)
            prompt += (
                f"\nVoiceover: a warm, calm narrator says in {lang_name}: "
                f"\"{scene.narration.strip()}\" "
                f"(clear voiceover speech, not on-screen text)"
            )
        return prompt


def _replace(src: Path, dst: Path) -> None:
    import shutil

    shutil.copyfile(src, dst)
