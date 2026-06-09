"""
ads/videos.py
-------------
[3단계] 컷별 첫 장면 이미지를 시작 프레임으로 비디오 클립 생성 + 결합.

기존 비디오 백엔드(app/backends — Sora/Veo/LTX)를 image-to-video 모드
(ClipSpec.first_frame)로 재사용한다.

길이 정책:
  백엔드가 보장하는 클립 길이는 supported_durations 뿐이다.
  스토리보드 컷은 보통 4~6초이므로,
    1) 컷마다 가장 가까운 지원 길이로 생성 요청하고
    2) 생성본이 컷 길이보다 길면 FFmpeg 로 앞부분만 트리밍한 뒤
    3) 전체를 하나의 광고 영상으로 결합한다.
  FFmpeg 가 없으면 트리밍/결합 없이 컷별 원본 클립만 제공한다(부분 성공).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..backends import ClipSpec, VideoBackend
from ..config import Settings
from ..pipeline.postprocess import (
    PostprocessError,
    append_outro,
    concat_clips,
    make_logo_outro,
)
from .prompts import build_video_prompt
from .schemas import AdStoryboard, AdStoryboardOptions, CutAsset

logger = logging.getLogger(__name__)

_RETRY_WAIT_SEC = 5.0


class VideosStageError(Exception):
    """비디오 단계 전체 실패. partial_assets 에 부분 산출물을 담는다."""

    def __init__(self, message: str, partial_assets: list[CutAsset]) -> None:
        super().__init__(message)
        self.partial_assets = partial_assets


async def run_videos_stage(
    settings: Settings,
    backend: VideoBackend,
    storyboard: AdStoryboard,
    options: AdStoryboardOptions,
    image_paths: dict[int, Path],
    out_dir: Path,
    text_exposure: str = "minimal",
    logo_outro: bool = False,
    logo_path: Optional[Path] = None,
) -> tuple[list[CutAsset], Optional[str]]:
    """
    컷별 비디오를 생성하고 최종 결합본을 만든다.

    Parameters
    ----------
    image_paths : {컷 번호: 첫 장면 이미지 경로} (2단계 산출물)

    Returns
    -------
    (컷별 결과 목록, 최종 결합본 경로 또는 None)

    Raises
    ------
    VideosStageError : 한 컷이라도 클립 생성에 실패한 경우(부분 산출물 동봉)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if not has_ffmpeg:
        logger.warning(
            "[ads/videos] FFmpeg 미설치: 트리밍/결합을 건너뛰고 "
            "컷별 원본 클립만 생성합니다."
        )

    # --- 1) 컷별 클립 생성 (이미지 → 비디오) ---------------------------
    assets: list[CutAsset] = []
    for cut in storyboard.cuts:
        asset = CutAsset(cut=cut.cut, status="in_progress")
        image_path = image_paths.get(cut.cut)
        if image_path is None or not Path(image_path).exists():
            asset.status = "failed"
            asset.error = f"컷 {cut.cut} 의 첫 장면 이미지가 없습니다."
            assets.append(asset)
            logger.warning("[ads/videos] ✗ %s", asset.error)
            continue

        prompt = build_video_prompt(cut, storyboard, locale=options.locale)
        asset.prompt_used = prompt
        spec = ClipSpec(
            prompt=prompt,
            duration_sec=float(cut.duration_sec),
            aspect_ratio=options.aspect_ratio,
            resolution=options.resolution,
            generate_audio=True,
            index=cut.cut,
            first_frame=Path(image_path),
            text_exposure=text_exposure,
        )
        gen_seconds = backend.normalize_duration(spec.duration_sec)
        out_path = out_dir / f"cut_{cut.cut:02d}.mp4"
        logger.info(
            "[ads/videos] 컷 %d 생성 시작 (요청 %d초 → 생성 %.0f초)",
            cut.cut, cut.duration_sec, gen_seconds,
        )

        try:
            result = await _generate_with_retry(
                backend, spec, out_path, retries=settings.clip_retries
            )
            clip_path = Path(result.path)
            asset.backend_job_id = result.meta.get("backend_job_id")

            # 컷 길이에 맞춰 트리밍(가능할 때만, 실패는 비치명).
            actual = result.duration_sec or gen_seconds
            if has_ffmpeg and cut.duration_sec < actual:
                trimmed = out_dir / f"cut_{cut.cut:02d}_trimmed.mp4"
                try:
                    await asyncio.to_thread(
                        _trim_clip, clip_path, trimmed, cut.duration_sec
                    )
                    clip_path = trimmed
                    logger.info(
                        "[ads/videos] 컷 %d 트리밍 (%.0f초 → %d초)",
                        cut.cut, actual, cut.duration_sec,
                    )
                except VideoTrimError as exc:
                    logger.warning(
                        "[ads/videos] 컷 %d 트리밍 실패(원본 사용): %s",
                        cut.cut, exc,
                    )

            asset.status = "completed"
            asset.path = str(clip_path)
            logger.info("[ads/videos] ✓ 컷 %d 완료 → %s", cut.cut, clip_path.name)
        except Exception as exc:  # noqa: BLE001 - 컷 단위 격리
            asset.status = "failed"
            asset.error = str(exc)
            logger.warning("[ads/videos] ✗ 컷 %d 실패: %s", cut.cut, exc)
        assets.append(asset)

    failed = [a for a in assets if a.status != "completed"]
    if failed:
        nums = ", ".join(str(a.cut) for a in failed)
        raise VideosStageError(
            f"컷 {nums} 비디오 생성에 실패했습니다. "
            f"첫 실패 사유: {failed[0].error}",
            partial_assets=assets,
        )

    # --- 2) 최종 결합 ---------------------------------------------------
    final_path: Optional[str] = None
    if has_ffmpeg:
        merged = out_dir / "final.mp4"
        clip_paths = [Path(a.path) for a in assets if a.path]
        try:
            await asyncio.to_thread(concat_clips, clip_paths, merged)
            final_path = str(merged)
            logger.info("[ads/videos] ✓ 최종 결합 완료 → %s", merged.name)
            # 로고 아웃트로(엔드카드) — opt-in. 모든 모델 공통 후처리.
            if logo_outro and logo_path and Path(logo_path).exists():
                try:
                    from ..llm.openai_llm import recommend_outro_background

                    context = " / ".join(
                        x for x in [storyboard.project, storyboard.concept] if x
                    ) or "advertisement"
                    bg = await recommend_outro_background(
                        settings, context, brand=storyboard.logo or "",
                        fallback=settings.logo_outro_bg_default,
                    )
                    outro = out_dir / "outro.mp4"
                    await asyncio.to_thread(
                        make_logo_outro,
                        Path(logo_path), merged, outro,
                        settings.logo_outro_duration_sec, bg,
                        settings.logo_outro_fade_sec,
                        settings.logo_outro_scale_ratio,
                    )
                    with_outro = out_dir / "final_outro.mp4"
                    await asyncio.to_thread(
                        append_outro, merged, outro, with_outro
                    )
                    final_path = str(with_outro)
                    logger.info("[ads/videos] ✓ 로고 아웃트로 추가(배경 %s)", bg)
                except Exception as exc:  # noqa: BLE001 - 비치명
                    logger.warning(
                        "[ads/videos] 아웃트로 추가 실패(본편 유지): %s", exc
                    )
        except PostprocessError as exc:
            # 클립은 모두 완성됐으므로 결합 실패는 부분 성공으로 처리한다.
            logger.warning(
                "[ads/videos] 최종 결합 실패(컷별 클립은 사용 가능): %s", exc
            )
    return assets, final_path


async def _generate_with_retry(
    backend: VideoBackend, spec: ClipSpec, out_path: Path, retries: int
):
    attempts = max(1, retries + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await backend.generate_clip(spec, out_path)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                logger.info(
                    "[ads/videos] 컷 %d 재시도 %d/%d (사유: %s)",
                    spec.index, attempt, attempts - 1, exc,
                )
                await asyncio.sleep(_RETRY_WAIT_SEC)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------- #
# FFmpeg 트리밍
# ---------------------------------------------------------------------- #
class VideoTrimError(Exception):
    """클립 트리밍 실패."""


def _trim_clip(src: Path, dst: Path, duration_sec: int) -> None:
    """클립 앞에서 duration_sec 초만 잘라 dst 로 저장한다(재인코딩)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoTrimError("ffmpeg 를 찾을 수 없습니다.")
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-t", str(duration_sec),
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise VideoTrimError(f"ffmpeg 실행 오류: {exc}") from exc
    if proc.returncode != 0 or not dst.exists():
        raise VideoTrimError(f"ffmpeg 트리밍 실패: {(proc.stderr or '')[-500:]}")
