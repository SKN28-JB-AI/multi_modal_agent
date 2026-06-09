"""
narration.py
------------
보이스오버/내레이션 '분량'을 영상(클립) 길이에 맞게 산정하는 공용 로직.

[배경 / 해결하려는 문제]
비디오 모델에 보이스오버를 맡기면, 영상 길이 대비 대사가 지나치게 짧아
'빈' 느낌이 났다(예: 6초 컷에 15자, 초당 2~3자). 원인은 스토리보드 단계에서
"씬당 한 문장, 6초=15자"로 분량을 과도하게 제한했기 때문이다.

이 모듈은 모델·언어마다 다른 '자연스럽지만 꽉 찬' 발화 속도를 기준으로,
클립 길이(초)에서 목표 발화 분량(글자/단어 수)을 계산한다.
  - CJK(한국어/일본어/중국어): 글자 수 기준(초당 char).
  - 그 외(영/스/불/독/베트남): 단어 수 기준(초당 word).

[설계 원칙]
- 외부 의존성 없음(순수 함수) -> 단위 테스트가 쉽고 부작용이 없다.
- '맞춤(per-model)'은 호출자가 백엔드가 실제로 만들어 내는 클립 길이
  (backend.normalize_duration 결과)를 duration_sec 으로 넘김으로써 달성된다.
  같은 스토리보드라도 Sora(4/8/12s)/Veo(4/6/8s)/LTX(6/8/10s)/Wan 등
  모델별 실제 클립 길이에 맞춰 분량이 달라진다.
- 속도 상수는 이 모듈 상단에서 조정할 수 있다(운영 튜닝 지점).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------- #
# 언어 표기 정규화 (서비스 다른 모듈과 동일 규칙)
# ---------------------------------------------------------------------- #
LANGUAGE_NAMES = {
    "ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German", "vi": "Vietnamese",
}

_LANGUAGE_ALIASES = {
    "korean": "ko", "english": "en", "japanese": "ja", "chinese": "zh",
    "spanish": "es", "french": "fr", "german": "de", "vietnamese": "vi",
}


def normalize_language(value):
    """'ko-KR' / 'korean' / 'KO' 등을 2글자 코드로 정규화. 비거나 모르면 'ko'."""
    if not value or not str(value).strip():
        return "ko"
    v = str(value).strip().lower().replace("_", "-")
    base = v.split("-", 1)[0]
    if base in LANGUAGE_NAMES:
        return base
    return _LANGUAGE_ALIASES.get(base, base or "ko")


def language_name(value):
    code = normalize_language(value)
    return LANGUAGE_NAMES.get(code, code.capitalize())


# ---------------------------------------------------------------------- #
# 발화 속도 모델 (운영 튜닝 지점)
# ---------------------------------------------------------------------- #
# '편안하지만 꽉 찬' 광고 내레이션 속도.
#   CJK: 초당 글자 수(characters/sec)
#   라틴 계열: 초당 단어 수(words/sec)
# 실제 자연 발화는 더 빠를 수 있으나, 비디오 모델이 또렷하게 발화하도록
# 약간 보수적으로 잡는다.
_CJK_CPS = {"ko": 5.0, "ja": 5.5, "zh": 5.0}
_LATIN_WPS = {"en": 2.5, "es": 2.6, "fr": 2.4, "de": 2.2, "vi": 3.0}
# 모르는 언어 기본값: CJK 글자 기준으로 보수적 처리.
_DEFAULT_RATE = 5.0
_DEFAULT_UNIT = "characters"

# 목표 분량 = 속도 x 길이 x 채움비율. 끝에 약간의 호흡만 남긴다.
FILL_TARGET = 0.92   # 목표(target)
FILL_LOW = 0.78      # 권장 하한
FILL_HIGH = 1.0      # 권장 상한(거의 꽉 채움)

# 대략적인 '문장 수' 환산(가독성 안내용).
_CHARS_PER_SENTENCE = 15
_WORDS_PER_SENTENCE = 9

# 한국어 안내문에서 쓸 단위 표기.
_UNIT_KO = {"characters": "자", "words": "단어"}


def _rate_for(language):
    """언어 -> (속도, 단위). 단위는 'characters' 또는 'words'."""
    code = normalize_language(language)
    if code in _CJK_CPS:
        return _CJK_CPS[code], "characters"
    if code in _LATIN_WPS:
        return _LATIN_WPS[code], "words"
    return _DEFAULT_RATE, _DEFAULT_UNIT


@dataclass(frozen=True)
class VoiceBudget:
    """클립 길이에 맞춘 보이스오버 분량 산정 결과."""

    language: str          # 정규화된 2글자 코드
    language_name: str     # 영어 언어명(프롬프트용)
    duration_sec: float    # 기준 클립 길이(초)
    unit: str              # 'characters' | 'words'
    rate: float            # 초당 단위 수
    target: int            # 목표 분량
    low: int               # 권장 하한
    high: int              # 권장 상한
    sentences: int         # 대략적인 권장 문장 수


def budget(duration_sec, language):
    """클립 길이(초)와 언어로 보이스오버 목표 분량을 계산한다."""
    code = normalize_language(language)
    rate, unit = _rate_for(code)
    d = max(1.0, float(duration_sec or 0.0))

    target = max(1, round(rate * d * FILL_TARGET))
    low = max(1, round(rate * d * FILL_LOW))
    high = max(low, round(rate * d * FILL_HIGH))

    per = _CHARS_PER_SENTENCE if unit == "characters" else _WORDS_PER_SENTENCE
    sentences = max(1, round(target / per))

    return VoiceBudget(
        language=code,
        language_name=LANGUAGE_NAMES.get(code, code.capitalize()),
        duration_sec=d,
        unit=unit,
        rate=rate,
        target=target,
        low=low,
        high=high,
        sentences=sentences,
    )


# ---------------------------------------------------------------------- #
# 프롬프트 클로즈 빌더 (비디오 모델에 넘길 영어 지시문)
# ---------------------------------------------------------------------- #
def _pace_phrase(b):
    """'길이를 꽉 채우는 자연스러운 속도' 영어 안내구."""
    plural = "s" if b.sentences != 1 else ""
    return (
        f"pace it to fill almost the entire {int(round(b.duration_sec))}s at a "
        f"natural, unhurried delivery (about {b.low}-{b.high} {b.unit}, "
        f"roughly {b.sentences} short sentence{plural}), keep speaking with "
        f"minimal silence and never rush or slur the words"
    )


def voiceover_line(text, duration_sec, language):
    """
    단일 씬 보이스오버 지시문(이미지/씬별 생성용).

    [중요] 기존 테스트/동작 호환을 위해 '... says in {Lang}: "{text}"' 형태의
    연속 부분 문자열을 반드시 포함한다(따옴표 대사 + 언어 + 화자 3요소).
    """
    b = budget(duration_sec, language)
    text = (text or "").strip()
    return (
        f"Voiceover: a warm, calm narrator delivers continuous spoken narration; "
        f"{_pace_phrase(b)}, and says in {b.language_name}: \"{text}\" "
        f"(spoken voiceover only, never on-screen text)"
    )


def voiceover_multi_line(spoken, duration_sec, language):
    """
    여러 씬을 한 클립으로 합치는 single 모드 보이스오버 지시문.

    [중요] 기존 테스트 호환을 위해 'speaks in {Lang}' 부분 문자열과
    호출자가 만든 spoken(따옴표 대사들)을 그대로 포함한다.
    """
    b = budget(duration_sec, language)
    return (
        f"Voiceover: a warm, calm narrator speaks in {b.language_name} across the "
        f"shots, {_pace_phrase(b)} total, in order: {spoken} "
        f"(clear continuous voiceover speech, not on-screen text)"
    )


def enhance_rule(duration_sec, language):
    """
    메시지 모드 프롬프트 변환(enhance) 시스템 지시문에 넣을 보이스오버 규칙.
    대사를 길이에 맞게 '꽉 채우되', 화면 텍스트로 새지 않게 한다.
    """
    b = budget(duration_sec, language)
    plural = "s" if b.sentences != 1 else ""
    return (
        f"If the idea implies narration or voiceover, write {b.language_name} "
        f"voiceover that fills almost the whole {int(round(b.duration_sec))}s at a "
        f"natural, unhurried pace - roughly {b.low}-{b.high} {b.unit} "
        f"({b.sentences} short sentence{plural}), with minimal silence and no "
        f"rushing. If the idea is purely visual with no implied speech, omit "
        f"voiceover entirely. Never place spoken words as on-screen text."
    )


def storyboard_hint(language):
    """
    스토리보드 생성 LLM 에 넣을 '내레이션 분량' 안내(한국어).
    각 컷/씬 길이를 거의 꽉 채우도록, 언어별 속도와 예시를 함께 제시한다.
    """
    rate, unit = _rate_for(language)
    lname = language_name(language)
    unit_ko = _UNIT_KO.get(unit, "자")
    ex6 = max(1, round(rate * 6 * FILL_TARGET))
    ex8 = max(1, round(rate * 8 * FILL_TARGET))
    return (
        f"내레이션/대사(voiceover/narration)는 각 컷(씬) 길이를 '거의 꽉 채우는' "
        f"분량으로 작성하세요. {lname} 기준 자연스러운 광고 내레이션 속도는 "
        f"초당 약 {rate:g}{unit_ko}입니다. 예) 6초 컷이면 약 {ex6}{unit_ko}, "
        f"8초면 약 {ex8}{unit_ko} 정도(여러 문장 가능). 끝에 약간의 호흡만 남기고 "
        f"침묵을 최소화하되, 너무 빠르게 뭉개지지 않게 하세요. 짧은 한 문장으로 "
        f"끝내지 마세요. 단, 화면에 직접 글자를 그리지는 말고(자막은 별도 처리) "
        f"'음성'으로만 전달되게 하세요."
    )
