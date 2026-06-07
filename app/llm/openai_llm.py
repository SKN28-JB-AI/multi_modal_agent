"""
llm/openai_llm.py
-----------------
OpenAI GPT 를 이용한 ① PDF 이해 / ② 스토리보드 변환.

[설계 노트]
- 비전 입력: PDF 페이지를 PNG 로 렌더링해 base64 data URI 로 전달한다.
  (텍스트 추출만으로는 레이아웃·이미지·무드보드를 놓치기 때문)
- 구조화 출력: response_format=json_object + pydantic 검증 + 1회 재시도.
  (json_schema strict 모드보다 모델 호환성이 넓다)
- 영상 프롬프트는 영어로 생성하게 한다(비디오 모델 성능이 가장 좋음).
  카피/내레이션은 요청 언어(기본 한국어)를 유지한다.
"""

from __future__ import annotations

import base64
import json
import logging

from pydantic import ValidationError

from ..config import Settings
from ..schemas import PdfJobOptions, Storyboard

logger = logging.getLogger(__name__)


_BRIEF_SYSTEM = """당신은 광고 기획서를 분석하는 전문 AD 플래너입니다.
주어진 광고 기획서(텍스트 + 페이지 이미지)를 분석해 JSON 으로 요약하세요.

반드시 아래 키를 가진 JSON 객체만 출력하세요:
{
  "product": "제품/서비스명과 핵심 특징",
  "brand": "브랜드명과 톤앤매너",
  "key_message": "광고의 핵심 메시지 한 문장",
  "target_audience": "타깃 고객",
  "visual_direction": "기획서에서 읽히는 비주얼 방향(색감/무드/레퍼런스)",
  "copy_candidates": ["기획서의 카피 문구들"],
  "constraints": "법적/브랜드 제약이 있다면",
  "notes": "그 외 영상 제작에 중요한 내용"
}"""

_STORYBOARD_SYSTEM = """당신은 광고 영상 디렉터입니다. 광고 브리프를 받아
AI 영상 생성 모델용 스토리보드를 JSON 으로 작성하세요.

규칙:
1. scenes 는 {max_scenes}개 이하, 각 씬 duration_sec 은 4~12초,
   전체 합이 약 {target_duration}초가 되게 하세요.
2. scene.prompt 는 **영어**로, 카메라 워크·조명·무드·동작을 구체적으로
   묘사하세요. 오디오(배경음/효과음) 묘사도 prompt 에 포함하세요.
3. 브랜드 로고, 실존 인물, 저작권 캐릭터, 화면 속 글자는 prompt 에
   넣지 마세요(AI 영상 모델이 거부하거나 깨뜨립니다).
4. scene.on_screen_text 는 {language} 로 작성하세요. 이 텍스트는 영상에
   직접 생성하지 않고 별도 자막(SRT)으로 처리됩니다.
5. narration_script 는 {language} 로, 전체 영상 길이에 맞는 분량으로.
6. 씬 간 비주얼 연속성을 위해 공통 스타일 키워드를 모든 prompt 에
   반복하세요(색감, 렌즈, 톤).
7. **내레이션 분배(중요)**: narration_script 를 씬별로 나눠 각 씬의
   narration 필드에 배치하세요. 씬의 narration 은 비디오 모델이
   보이스오버로 **직접 발화**하므로 반드시 채워야 합니다.
   - {language} 로, 씬당 한 문장.
   - 발화 가능 분량 엄수: 6초 씬 기준 15자 내외(초당 2~3자).
     길면 잘리거나 뭉개집니다.
   - 브리프에 내레이션/카피 제안이 있으면 그것을 우선 사용해 분배하세요.

반드시 아래 구조의 JSON 객체만 출력하세요:
{{
  "title": "...", "summary": "...", "target_audience": "...",
  "narration_script": "...",
  "scenes": [
    {{"index": 0, "prompt": "...", "duration_sec": 6,
      "audio_description": "...", "narration": "...",
      "on_screen_text": "..."}}
  ]
}}"""


class OpenAILLM:
    """PDF 이해/스토리보드 변환용 OpenAI LLM 래퍼."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "PDF 모드에는 OPENAI_API_KEY 가 필요합니다(파싱/스토리보드 LLM)."
            )
        from openai import AsyncOpenAI

        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_llm_model

    # ------------------------------------------------------------------ #
    # ① PDF 이해 → 브리프(JSON dict)
    # ------------------------------------------------------------------ #
    async def analyze_pdf(self, text: str, page_images: list[bytes]) -> dict:
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "다음은 광고 기획서입니다. 페이지 이미지와 추출 텍스트를 "
                    "함께 분석하세요.\n\n[추출 텍스트]\n" + text[:30000]
                ),
            }
        ]
        for img in page_images:
            b64 = base64.b64encode(img).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        resp = await self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _BRIEF_SYSTEM},
                {"role": "user", "content": content},
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")

    # ------------------------------------------------------------------ #
    # ② 브리프/메시지 → 스토리보드
    # ------------------------------------------------------------------ #
    async def make_storyboard(
        self, brief: dict, options: PdfJobOptions
    ) -> Storyboard:
        system = _STORYBOARD_SYSTEM.format(
            max_scenes=options.max_scenes,
            target_duration=int(options.target_total_duration_sec),
            language=options.language,
        )
        user = (
            "광고 브리프:\n"
            + json.dumps(brief, ensure_ascii=False, indent=2)
        )

        last_error: Exception | None = None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # 검증 실패 시 오류를 알려주고 1회 재생성
        for _ in range(2):
            resp = await self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=messages,
            )
            raw = resp.choices[0].message.content or "{}"
            try:
                return Storyboard.model_validate_json(raw)
            except ValidationError as exc:
                last_error = exc
                logger.warning("스토리보드 검증 실패, 재시도: %s", exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "출력이 스키마 검증에 실패했습니다. 오류를 고쳐 "
                            f"다시 JSON 만 출력하세요:\n{exc}"
                        ),
                    }
                )
        raise RuntimeError(f"스토리보드 생성 실패(스키마 불일치): {last_error}")
