"""
pipeline/outro.py
-----------------
로고 아웃트로(엔드카드) 클립 빌더 — v1(영상 생성)·v2(광고 파이프라인) 공용.

순서:
  1) 스타일화: 지정 로고 이미지를 OpenAI images.edit 로 영상 프롬프트
     분위기에 맞는 풀프레임 엔드카드 이미지로 변환 → make_image_outro.
  2) 폴백: 스타일화 불가(비활성/키 없음/호출 실패) 시 기존 방식 —
     LLM 배경색 추천 + 단색 배경 중앙 로고(make_logo_outro).

두 경로 모두 본편 해상도/fps/오디오 구성에 맞춰 생성되므로 concat 안전.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..llm.openai_llm import recommend_outro_background, stylize_logo_endcard
from . import postprocess

logger = logging.getLogger(__name__)


async def build_outro(
    settings,
    logo_path: Path,
    context: str,
    reference_video: Path,
    out_dir: Path,
    aspect_ratio: str = "16:9",
    brand: str = "",
) -> Path:
    """아웃트로 클립(outro.mp4)을 만들어 경로를 반환한다."""
    outro = out_dir / "outro.mp4"

    # 1) 스타일화 엔드카드(영상 분위기 반영) 우선
    card = out_dir / "outro_card.png"
    stylized = await stylize_logo_endcard(
        settings, logo_path, context, card, aspect_ratio=aspect_ratio
    )
    if stylized is not None:
        await asyncio.to_thread(
            postprocess.make_image_outro,
            stylized, reference_video, outro,
            settings.logo_outro_duration_sec, settings.logo_outro_fade_sec,
        )
        logger.info("아웃트로: 스타일화 엔드카드 사용(%s)", stylized.name)
        return outro

    # 2) 폴백: 단색 배경 + 중앙 로고 (기존 방식)
    bg = await recommend_outro_background(
        settings, context, brand=brand,
        fallback=settings.logo_outro_bg_default,
    )
    await asyncio.to_thread(
        postprocess.make_logo_outro,
        logo_path, reference_video, outro,
        settings.logo_outro_duration_sec, bg,
        settings.logo_outro_fade_sec, settings.logo_outro_scale_ratio,
    )
    logger.info("아웃트로: 단색 배경 폴백 사용(배경 %s)", bg)
    return outro
