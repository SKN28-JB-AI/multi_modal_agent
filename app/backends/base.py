"""
backends/base.py
----------------
비디오 생성 백엔드 추상화 + 레지스트리.

[확장 방법]
1) VideoBackend 를 상속해 generate_clip() 을 구현한다.
2) register("모델이름")(클래스, **고정파라미터) 로 등록한다.
   - 같은 클래스를 다른 파라미터로 여러 이름에 등록할 수 있다.
     (예: sora-2 / sora-2-pro 는 같은 SoraBackend, model 파라미터만 다름)
3) 끝. /v1/models 와 요청의 model 필드에 자동으로 노출된다.

[규약]
- generate_clip 은 완성된 MP4 를 out_path 에 저장해야 한다(영상+오디오).
- duration 은 백엔드마다 지원 값이 다르므로 normalize_duration 으로
  가장 가까운 지원 값으로 보정해서 사용한다.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from ..config import Settings


# ---------------------------------------------------------------------- #
# 글자 깨짐 방지 + 단계별 글자 노출 (text-exposure policy)
# ---------------------------------------------------------------------- #
# AI 비디오 모델은 화면 글자(특히 한글 등 비라틴)를 거의 항상 깨뜨린다.
# 그렇다고 글자를 완전히 없애면 광고가 어색해지므로, '모델이 직접 그리는
# 글자'의 허용 수준을 4단계로 제어한다. 또렷하게 읽혀야 하는 한글 카피·자막·
# 로고는 모델이 아니라 후처리(자막 번인/오버레이, 실폰트)로 입히는 것을 권장한다.
#
#   none     : 화면 글자 완전 금지(가장 깔끔하나 밋밋할 수 있음)
#   minimal  : 또렷한 카피/자막/UI/워터마크는 금지하되, 배경 간판·제품 표면 등
#              장면에 자연스러운 '환경 텍스트'는 흐릿하게 허용(기본값)
#   moderate : 짧고 명료한 라틴 글자/숫자/간판은 허용. 단 한글 등 비라틴은
#              여전히 억제(깨짐 방지)
#   full     : 제한 없음(모델 자유, 깨짐 위험은 사용자 책임)
#
# (대사/내레이션은 '음성'이며 화면 글자가 아니므로 모든 단계에서 영향 없음.)

TEXT_EXPOSURE_LEVELS = ("none", "minimal", "moderate", "full")
DEFAULT_TEXT_EXPOSURE = "minimal"

# 단계별 '긍정 프롬프트' 클로즈(full 은 빈 문자열).
_TEXT_CLAUSES: dict[str, str] = {
    "none": (
        "Absolutely no on-screen text, letters, words, captions, subtitles, "
        "numbers, signage, UI, watermark, or logo of any kind. Purely visual, "
        "text-free footage; any wording is delivered only as spoken voiceover."
    ),
    "minimal": (
        "Do not add overlaid captions, subtitles, title cards, large readable "
        "text, watermark, logo, or UI. Only incidental, out-of-focus background "
        "or environmental text (distant signage, product surfaces) may appear "
        "naturally and must never be a focal element. Never render Korean or any "
        "non-Latin script."
    ),
    "moderate": (
        "Avoid garbled or distorted lettering. Short, clear Latin words, numbers, "
        "or natural signage may appear when they fit the scene, but do not render "
        "Korean or other non-Latin scripts (those are added in post-production)."
    ),
    "full": "",
}

# 단계별 negative_prompt 금지어(full 은 빈 문자열 → 제약 없음).
_TEXT_NEGATIVES: dict[str, str] = {
    "none": (
        "text, letters, words, captions, subtitles, title cards, typography, "
        "writing, signage, watermark, logo, gibberish text, distorted text, "
        "garbled characters, hangul text, on-screen text, UI elements"
    ),
    "minimal": (
        "captions, subtitles, title cards, large on-screen text, watermark, logo, "
        "UI elements, gibberish text, distorted text, garbled characters, "
        "hangul text, korean text, non-latin text"
    ),
    "moderate": (
        "gibberish text, distorted text, garbled characters, malformed letters, "
        "hangul text, korean text, non-latin text"
    ),
    "full": "",
}

# 하위호환 별칭(기존 import/none 단계).
NO_TEXT_CLAUSE = _TEXT_CLAUSES["none"]
NO_TEXT_NEGATIVE = _TEXT_NEGATIVES["none"]


def normalize_text_exposure(level: str | None) -> str:
    """글자 노출 단계 문자열을 검증·정규화한다(모르면 기본값)."""
    if not level:
        return DEFAULT_TEXT_EXPOSURE
    v = str(level).strip().lower()
    return v if v in TEXT_EXPOSURE_LEVELS else DEFAULT_TEXT_EXPOSURE


def text_clause_for(level: str | None) -> str:
    """단계에 해당하는 긍정 프롬프트 클로즈(full 은 '')."""
    return _TEXT_CLAUSES[normalize_text_exposure(level)]


def negative_for(level: str | None) -> str:
    """단계에 해당하는 negative 금지어(full 은 '')."""
    return _TEXT_NEGATIVES[normalize_text_exposure(level)]


def apply_text_policy(prompt: str, level: str | None = None) -> str:
    """긍정 프롬프트에 단계별 글자 노출 클로즈를 1회 부착한다."""
    clause = text_clause_for(level)
    base = (prompt or "").rstrip()
    if not clause or clause in base:
        return base
    sep = "\n" if base else ""
    return f"{base}{sep}{clause}"


def negative_prompt_for(level: str | None, existing: str | None = None) -> str:
    """기존 negative 에 단계별 금지어를 합친다(full 이면 기존만 반환)."""
    neg = negative_for(level)
    existing = (existing or "").strip()
    if not neg:
        return existing
    if not existing:
        return neg
    if neg in existing:
        return existing
    return f"{existing}, {neg}"


# --- 하위호환 헬퍼(none 단계로 동작) ---
def with_no_text_clause(prompt: str) -> str:
    """[deprecated] none 단계 클로즈 부착. apply_text_policy(p, 'none') 와 동일."""
    return apply_text_policy(prompt, "none")


def merge_negative_prompt(existing: str | None) -> str:
    """[deprecated] none 단계 negative 병합. negative_prompt_for('none', e) 와 동일."""
    return negative_prompt_for("none", existing)


@dataclass
class ClipSpec:
    """클립 1개 생성 사양 (백엔드 공통 입력)."""

    prompt: str
    duration_sec: float = 6.0
    aspect_ratio: str = "16:9"     # "16:9" | "9:16"
    resolution: str = "1080p"      # "720p" | "1080p"
    generate_audio: bool = True
    index: int = 0                 # 씬 번호(로그/파일명용)
    # image-to-video: 시작 프레임 이미지 경로(지원 백엔드만 사용).
    # None 이면 기존과 동일한 text-to-video 로 동작한다.
    first_frame: Optional[Path] = None
    # 화면 글자 노출 단계: none / minimal / moderate / full (DEFAULT_TEXT_EXPOSURE).
    # 모델이 직접 그리는 글자의 허용 수준을 제어한다(또렷한 한글은 후처리 권장).
    text_exposure: str = "minimal"


@dataclass
class ClipResult:
    """클립 생성 결과."""

    path: Path
    duration_sec: float
    meta: dict = field(default_factory=dict)   # 백엔드별 부가 정보(잡ID 등)


class BackendNotConfigured(Exception):
    """필요한 API 키가 없어 백엔드를 사용할 수 없음."""


class ClipGenerationError(Exception):
    """클립 생성 실패."""


class VideoBackend(abc.ABC):
    """비디오 생성 백엔드 인터페이스."""

    provider: str = ""
    description: str = ""
    # 지원하는 클립 길이(초). normalize_duration 이 참조한다.
    supported_durations: tuple[float, ...] = (6.0,)
    # 기존 생성물 부분 수정(remix) 지원 여부. 지원 백엔드만 True 로 재정의.
    supports_remix: bool = False
    # 시작 프레임 이미지 입력(image-to-video) 지원 여부.
    supports_image_input: bool = False
    # 네이티브 오디오(보이스오버/효과음) 생성 지원 여부.
    # False 인 모델(예: 무음 wan-2.2)은 임베디드 보이스오버가 무의미하므로
    # 호출자가 이 플래그로 판단해 임베디드 내레이션을 생략한다.
    supports_audio: bool = True

    def __init__(self, settings: Settings, **params) -> None:
        self.settings = settings
        self.params = params

    # -------------------------------------------------------------- #
    @abc.abstractmethod
    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        """spec 대로 클립을 생성해 out_path(MP4)에 저장한다."""

    @classmethod
    @abc.abstractmethod
    def is_configured(cls, settings: Settings) -> bool:
        """이 백엔드 사용에 필요한 API 키가 설정되어 있는가."""

    # -------------------------------------------------------------- #
    async def remix_clip(
        self, source_video_id: str, prompt: str, out_path: Path
    ) -> ClipResult:
        """
        기존 생성물(source_video_id)을 프롬프트로 부분 수정한다.
        supports_remix=True 인 백엔드만 구현한다.
        """
        raise ClipGenerationError(
            f"'{self.__class__.__name__}' 백엔드는 remix 를 지원하지 않습니다."
        )

    # -------------------------------------------------------------- #
    def normalize_duration(self, requested: float) -> float:
        """요청 길이를 백엔드가 지원하는 가장 가까운 값으로 보정."""
        return min(self.supported_durations, key=lambda d: abs(d - requested))

    def audio_supported(self) -> bool:
        """이 인스턴스가 네이티브 오디오를 생성하는가(등록 파라미터 우선)."""
        return bool(self.params.get("supports_audio", self.supports_audio))


# ---------------------------------------------------------------------- #
# 레지스트리
# ---------------------------------------------------------------------- #
@dataclass
class _Registration:
    backend_cls: Type[VideoBackend]
    params: dict


_REGISTRY: dict[str, _Registration] = {}


def register(name: str, backend_cls: Type[VideoBackend], **params) -> None:
    """백엔드를 모델 이름으로 등록한다."""
    _REGISTRY[name] = _Registration(backend_cls=backend_cls, params=params)


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def available_models() -> list[str]:
    return list(_REGISTRY.keys())


def get_backend(name: str, settings: Settings) -> VideoBackend:
    """등록된 백엔드 인스턴스를 만든다. 미등록이면 KeyError."""
    reg = _REGISTRY.get(name)
    if reg is None:
        raise KeyError(
            f"등록되지 않은 모델입니다: '{name}'. "
            f"사용 가능: {', '.join(sorted(_REGISTRY))}"
        )
    if not reg.backend_cls.is_configured(settings):
        raise BackendNotConfigured(
            f"'{name}' 백엔드에 필요한 API 키가 설정되지 않았습니다. "
            f".env 를 확인하세요."
        )
    return reg.backend_cls(settings, **reg.params)


def backend_info(settings: Settings) -> list[dict]:
    """GET /v1/models 응답용 백엔드 메타데이터."""
    infos = []
    for name, reg in sorted(_REGISTRY.items()):
        cls = reg.backend_cls
        instance_durations = reg.params.get(
            "supported_durations", cls.supported_durations
        )
        infos.append(
            {
                "name": name,
                "provider": cls.provider,
                "description": cls.description,
                "configured": cls.is_configured(settings),
                "supported_durations": [float(d) for d in instance_durations],
                "supports_remix": cls.supports_remix,
                "supports_image_input": cls.supports_image_input,
                "supports_audio": bool(
                    reg.params.get("supports_audio", cls.supports_audio)
                ),
            }
        )
    return infos
