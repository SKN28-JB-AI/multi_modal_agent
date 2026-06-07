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
        storyboard = Storyboard(
            title="message",
            scenes=[
                Scene(
                    index=0,
                    prompt=req["prompt"],
                    duration_sec=req.get("duration_sec") or 6.0,
                )
            ],
        )
        self.manager.update(job, storyboard=storyboard)
        await self._generate_and_finish(job, storyboard, options=None)

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
        await self._generate_and_finish(job, storyboard, options, logo_path)

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

        semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_clips))
        results: dict[int, Path] = {}
        durations: dict[int, float] = {}
        done_count = 0
        progress_lock = asyncio.Lock()

        async def _one_scene(scene: Scene, state: SceneState) -> None:
            nonlocal done_count
            spec = ClipSpec(
                prompt=self._compose_prompt(scene),
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

        # --- ④ 후처리 -------------------------------------------------- #
        self.manager.update(
            job, status=JobStatus.POSTPROCESSING, progress=0.9,
            scenes=scene_states,
        )
        ordered_clips = [results[s.index] for s in scenes]
        ordered_durations = [durations[s.index] for s in scenes]
        current = job_dir / "merged.mp4"
        await asyncio.to_thread(postprocess.concat_clips, ordered_clips, current)

        # SRT (씬 카피가 있을 때만)
        srt_path = job_dir / "subtitles.srt"
        has_srt = await asyncio.to_thread(
            postprocess.write_srt, storyboard, ordered_durations, srt_path
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
                postprocess.overlay_logo, current, logo_path, branded
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
    def _compose_prompt(scene: Scene) -> str:
        """씬 프롬프트 + 오디오 묘사를 합성한다."""
        prompt = scene.prompt.strip()
        if scene.audio_description.strip():
            prompt += f"\nAudio: {scene.audio_description.strip()}"
        return prompt


def _replace(src: Path, dst: Path) -> None:
    import shutil

    shutil.copyfile(src, dst)
