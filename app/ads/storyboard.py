"""
ads/storyboard.py
-----------------
[1단계] 입력 프롬프트 → 스토리보드 JSON 생성.

OpenAI chat.completions.parse(구조화 출력, Pydantic 스키마)를 1차로 시도하고,
모델이 json_schema 형식을 지원하지 않으면 json_object 모드로 폴백한다.
생성 결과는 normalize_storyboard 로 후처리하여 컷 번호/타임코드/총 길이의
일관성을 보장한다.
"""

from __future__ import annotations

import asyncio
import json

from ..config import Settings
from .prompts import build_storyboard_messages
from .schemas import AdStoryboard, AdStoryboardOptions


class StoryboardGenerationError(Exception):
    """스토리보드 생성 실패."""


def normalize_storyboard(sb: AdStoryboard, options: AdStoryboardOptions) -> AdStoryboard:
    """
    LLM 출력의 사소한 불일치를 교정한다.
      - 컷 번호를 1..N 으로 재부여
      - duration 합이 목표와 다르면 마지막 컷에서 보정(최소 1초 보장)
      - timecode 를 duration 기준으로 재계산
      - aspect_ratio 를 요청 옵션으로 고정
    """
    if not sb.cuts:
        raise StoryboardGenerationError("스토리보드에 컷이 없습니다.")

    cuts = sorted(sb.cuts, key=lambda c: c.cut)

    for c in cuts:
        if c.duration_sec < 1:
            c.duration_sec = 1
    diff = options.total_duration_sec - sum(c.duration_sec for c in cuts)
    if diff != 0:
        last = cuts[-1]
        last.duration_sec = max(1, last.duration_sec + diff)

    def _fmt(sec: int) -> str:
        return f"{sec // 60:02d}:{sec % 60:02d}"

    elapsed = 0
    for idx, c in enumerate(cuts, start=1):
        c.cut = idx
        c.timecode = f"{_fmt(elapsed)}-{_fmt(elapsed + c.duration_sec)}"
        elapsed += c.duration_sec

    sb.cuts = cuts
    sb.total_duration_sec = sum(c.duration_sec for c in cuts)
    sb.aspect_ratio = options.aspect_ratio
    if not sb.logo:
        sb.logo = options.brand
    return sb


async def generate_storyboard(
    settings: Settings,
    prompt: str,
    options: AdStoryboardOptions,
) -> AdStoryboard:
    """프롬프트로 스토리보드를 생성한다. 실패 시 StoryboardGenerationError."""
    messages = build_storyboard_messages(
        prompt=prompt,
        cut_count=options.cut_count,
        total_duration_sec=options.total_duration_sec,
        aspect_ratio=options.aspect_ratio,
        locale=options.locale,
        brand=options.brand,
    )
    sb = await asyncio.to_thread(
        _call_structured, settings, settings.ad_storyboard_model, messages
    )
    return normalize_storyboard(sb, options)


def _call_structured(settings: Settings, model: str,
                     messages: list[dict]) -> AdStoryboard:
    """구조화 출력 1차 시도 → json_object 폴백 → 실패 시 예외."""
    from openai import OpenAI

    if not settings.openai_api_key:
        raise StoryboardGenerationError(
            "OPENAI_API_KEY 가 설정되지 않아 스토리보드를 생성할 수 없습니다."
        )
    client = OpenAI(api_key=settings.openai_api_key)

    # --- 1차: Pydantic 스키마 기반 구조화 출력 -------------------------
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=AdStoryboard,
        )
        message = completion.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise StoryboardGenerationError(f"모델이 생성을 거부했습니다: {refusal}")
        if message.parsed is not None:
            return message.parsed
        if message.content:
            return AdStoryboard.model_validate(json.loads(message.content))
        raise StoryboardGenerationError("구조화 출력이 비어 있습니다.")
    except StoryboardGenerationError:
        raise
    except Exception as parse_exc:  # noqa: BLE001 - 폴백으로 넘어간다.
        first_error = parse_exc

    # --- 2차 폴백: json_object 모드 ------------------------------------
    try:
        schema_hint = json.dumps(
            AdStoryboard.model_json_schema(), ensure_ascii=False
        )
        fallback_messages = list(messages) + [
            {
                "role": "system",
                "content": (
                    "다음 JSON Schema 를 정확히 따르는 JSON 객체만 출력하세요. "
                    "설명 문장이나 코드 블록 없이 JSON 만 출력합니다.\n"
                    f"{schema_hint}"
                ),
            }
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=fallback_messages,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        return AdStoryboard.model_validate(json.loads(content))
    except Exception as fallback_exc:  # noqa: BLE001
        raise StoryboardGenerationError(
            "스토리보드 생성에 실패했습니다. "
            f"(구조화 출력 오류: {first_error} / 폴백 오류: {fallback_exc})"
        ) from fallback_exc
