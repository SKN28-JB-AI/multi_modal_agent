"""
ads/prompts.py
--------------
단계·모델별 프롬프트 빌더.

설계 원칙
  - 같은 단계라도 사용하는 모델에 따라 효과적인 프롬프트 형식이 다르다.
    예) gpt-image 계열은 긴 서술형 묘사에 강하고, FLUX 는 키워드 중심의
        간결한 문장이 잘 듣는다.
  - "모델 패밀리 → 빌더 함수" 매핑으로 이를 흡수한다. 새 모델 추가 시
    빌더만 등록하면 서비스 코드는 그대로다.

주의(도메인 제약)
  - Sora 의 input_reference 이미지는 사람 얼굴이 포함되면 거부될 수 있다.
    → 모든 이미지 프롬프트에 얼굴 비노출(뒷모습/실루엣/원거리) 지시를 넣는다.
  - 이미지에 글자를 구워 넣으면 비디오 단계에서 어색하게 정지되어 보인다.
    → 자막/로고/텍스트 렌더링 금지 지시를 넣는다.
"""

from __future__ import annotations

from typing import Callable

from .schemas import AdStoryboard, Cut

# ====================================================================== #
# 1) 스토리보드 생성 (LLM)
# ====================================================================== #
_STORYBOARD_SYSTEM = """\
당신은 금융권 브랜드 광고를 전문으로 하는 크리에이티브 디렉터입니다.
사용자의 프롬프트를 바탕으로 영상 광고 스토리보드를 작성합니다.

규칙:
- 반드시 {locale} 언어로 작성합니다.
- 컷 수는 정확히 {cut_count}개, 전체 길이는 정확히 {total_duration_sec}초입니다.
- 각 컷의 duration_sec 합계가 total_duration_sec 와 일치해야 합니다.
- timecode 는 "MM:SS-MM:SS" 형식으로 컷 순서대로 빈틈없이 이어집니다.
- scene/visual 은 영상 생성 AI 가 그대로 그릴 수 있도록 구체적으로 묘사합니다
  (장소, 인물(얼굴이 화면에 정면 노출되지 않는 연출 권장), 행동, 색감, 조명).
- sfx 에는 효과음·음악 큐를, voiceover 에는 성우 대사를 적습니다(없으면 빈 문자열).
- 마지막 컷은 브랜드({brand}) 로고 리빌과 CTA(행동 유도)로 마무리합니다.
- 금융 광고이므로 과장·허위 수익 약속 표현은 쓰지 않습니다.
"""

_STORYBOARD_USER = """\
다음 프롬프트로 광고 비디오 스토리보드를 작성해 주세요.

프롬프트:
{prompt}

제작 조건:
- 컷 수: {cut_count}
- 전체 길이: {total_duration_sec}초
- 화면 비율: {aspect_ratio}
- 브랜드 표기(logo 필드): {brand}
"""


def build_storyboard_messages(
    prompt: str,
    cut_count: int,
    total_duration_sec: int,
    aspect_ratio: str,
    locale: str = "ko-KR",
    brand: str = "JB금융그룹",
) -> list[dict]:
    """스토리보드 생성용 chat messages."""
    system = _STORYBOARD_SYSTEM.format(
        locale=locale,
        cut_count=cut_count,
        total_duration_sec=total_duration_sec,
        brand=brand,
    )
    user = _STORYBOARD_USER.format(
        prompt=prompt,
        cut_count=cut_count,
        total_duration_sec=total_duration_sec,
        aspect_ratio=aspect_ratio,
        brand=brand,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ====================================================================== #
# 2) 컷 첫 장면 이미지 생성 (이미지 모델별 분기)
# ====================================================================== #
# 공통 제약: 비디오 첫 프레임(특히 Sora input_reference) 호환.
_IMAGE_CONSTRAINTS = (
    "No human faces visible (use back view, silhouette, or distant figures only). "
    "Do not render any text, captions, subtitles, logos, or watermarks. "
    "Clean cinematic still frame suitable as the first frame of a video."
)


def _image_prompt_descriptive(cut: Cut, sb: AdStoryboard) -> str:
    """gpt-image / imagen 계열: 풍부한 서술형 프롬프트가 효과적."""
    mood = ", ".join(sb.mood) if sb.mood else "cinematic"
    return (
        f"A cinematic first frame for a commercial film.\n"
        f"Campaign concept: {sb.concept}\n"
        f"Scene: {cut.scene}\n"
        f"Visual details: {cut.visual}\n"
        f"Camera: {cut.camera}\n"
        f"Mood keywords: {mood}\n"
        f"High production value, professional color grading, "
        f"aspect ratio {sb.aspect_ratio} framing.\n"
        f"{_IMAGE_CONSTRAINTS}"
    )


def _image_prompt_concise(cut: Cut, sb: AdStoryboard) -> str:
    """FLUX / dall-e 계열: 짧고 명료한 한 문단 + 핵심 제약."""
    mood = ", ".join(sb.mood[:3]) if sb.mood else "cinematic"
    return (
        f"Cinematic commercial still frame: {cut.scene}. {cut.visual} "
        f"Camera: {cut.camera}. Mood: {mood}. {_IMAGE_CONSTRAINTS}"
    )


_IMAGE_PROMPT_BUILDERS: dict[str, Callable[[Cut, AdStoryboard], str]] = {
    "openai": _image_prompt_descriptive,
    "gemini": _image_prompt_descriptive,
    "qwen": _image_prompt_descriptive,   # Qwen-Image 도 서술형 프롬프트에 강함
    "fal": _image_prompt_concise,
}


def build_image_prompt(provider_family: str, cut: Cut, sb: AdStoryboard) -> str:
    """이미지 모델 패밀리('openai'|'gemini'|'qwen'|'fal')에 맞는 첫 장면 프롬프트."""
    builder = _IMAGE_PROMPT_BUILDERS.get(provider_family, _image_prompt_descriptive)
    return builder(cut, sb)


# ====================================================================== #
# 3) 컷 비디오 생성 (이미지 → 비디오)
# ====================================================================== #
def build_video_prompt(cut: Cut, sb: AdStoryboard, locale: str = "ko-KR") -> str:
    """
    첫 프레임 이미지가 주어지는 image-to-video 프롬프트.
    장면·카메라·사운드(대사/효과음/음악)를 함께 서술하면 오디오 동기화
    품질이 올라간다(기존 운영 경험 반영).
    """
    lang = "Korean" if locale.lower().startswith("ko") else locale
    mood = ", ".join(sb.mood) if sb.mood else ""
    lines = [
        f"{cut.duration_sec}-second commercial shot starting exactly from "
        f"the provided first frame. {cut.scene}",
        f"Motion and visual: {cut.visual}",
        f"Camera: {cut.camera}",
    ]
    if mood:
        lines.append(f"Mood: {mood}.")
    if sb.music and sb.music.genre:
        lines.append(
            f"Music: {sb.music.genre}, {sb.music.bpm} BPM, beat-synced editing."
        )
    if cut.sfx:
        lines.append(f"Sound effects: {cut.sfx}")
    if cut.voiceover:
        lines.append(f'{lang} voiceover says: "{cut.voiceover}"')
    if cut.transition:
        lines.append(f"Ends with: {cut.transition}.")
    return "\n".join(lines)
