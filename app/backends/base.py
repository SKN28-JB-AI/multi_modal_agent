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
            }
        )
    return infos
