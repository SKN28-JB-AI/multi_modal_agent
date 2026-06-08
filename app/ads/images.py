"""
ads/images.py
-------------
[2단계] 스토리보드의 컷별 '첫 장면' 이미지 생성 실행기.

흐름(컷마다):
  1) 모델 패밀리별 프롬프트 빌드 (prompts.build_image_prompt)
  2) 이미지 백엔드 호출 → 원본 바이트
  3) 비디오 목표 해상도로 cover-crop 리사이즈 후 PNG 저장
     (Sora input_reference 는 비디오 해상도와 픽셀 단위 일치가 필요)

개별 컷 실패는 그 컷만 failed 로 기록하고 다음 컷을 계속 진행한다.
전체 단계의 성공 조건은 '모든 컷 이미지 생성 성공'이다
(하나라도 빠지면 3단계 비디오가 불가능하므로).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..config import Settings
from .image_backends import (
    ImageBackend,
    ImageGenerationError,
    ImageSpec,
    cover_resize,
    target_pixel_size,
)
from .prompts import build_image_prompt
from .schemas import AdStoryboard, AdStoryboardOptions, CutAsset

logger = logging.getLogger(__name__)

_RETRY_WAIT_SEC = 3.0


class ImagesStageError(Exception):
    """이미지 단계 전체 실패. partial_assets 에 부분 산출물을 담는다."""

    def __init__(self, message: str, partial_assets: list[CutAsset]) -> None:
        super().__init__(message)
        self.partial_assets = partial_assets


async def run_images_stage(
    settings: Settings,
    backend: ImageBackend,
    storyboard: AdStoryboard,
    options: AdStoryboardOptions,
    out_dir: Path,
) -> list[CutAsset]:
    """
    모든 컷의 첫 장면 이미지를 생성해 out_dir 에 저장한다.

    Returns
    -------
    list[CutAsset] : 컷별 결과(성공 시 path 채워짐)

    Raises
    ------
    ImagesStageError : 한 컷이라도 최종 실패한 경우(부분 산출물 동봉)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = target_pixel_size(options.aspect_ratio, options.resolution)

    assets: list[CutAsset] = []
    for cut in storyboard.cuts:
        asset = CutAsset(cut=cut.cut, status="in_progress")
        prompt = build_image_prompt(backend.family, cut, storyboard)
        asset.prompt_used = prompt
        out_path = out_dir / f"cut_{cut.cut:02d}.png"

        try:
            raw = await _generate_with_retry(
                backend,
                ImageSpec(prompt=prompt, aspect_ratio=options.aspect_ratio,
                          index=cut.cut),
                retries=settings.image_retries,
            )
            # cover_resize 는 PIL 디코드를 겸하므로 손상 바이트도 여기서 걸러진다.
            await asyncio.to_thread(cover_resize, raw, width, height, out_path)
            asset.status = "completed"
            asset.path = str(out_path)
            logger.info("[ads/images] ✓ 컷 %d 저장 → %s", cut.cut, out_path.name)
        except Exception as exc:  # noqa: BLE001 - 컷 단위 격리
            asset.status = "failed"
            asset.error = str(exc)
            logger.warning("[ads/images] ✗ 컷 %d 실패: %s", cut.cut, exc)
        assets.append(asset)

    failed = [a for a in assets if a.status != "completed"]
    if failed:
        nums = ", ".join(str(a.cut) for a in failed)
        raise ImagesStageError(
            f"컷 {nums} 이미지 생성에 실패했습니다. "
            f"첫 실패 사유: {failed[0].error}",
            partial_assets=assets,
        )
    return assets


async def _generate_with_retry(
    backend: ImageBackend, spec: ImageSpec, retries: int
) -> bytes:
    """일시 오류 대비 재시도. 마지막 예외를 그대로 올린다."""
    attempts = max(1, retries + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = await backend.generate_image(spec)
            if not raw:
                raise ImageGenerationError("이미지 응답이 비어 있습니다.")
            return raw
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                logger.info(
                    "[ads/images] 컷 %d 재시도 %d/%d (사유: %s)",
                    spec.index, attempt, attempts - 1, exc,
                )
                await asyncio.sleep(_RETRY_WAIT_SEC)
    assert last_exc is not None
    raise last_exc
