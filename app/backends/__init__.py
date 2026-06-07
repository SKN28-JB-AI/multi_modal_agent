"""
backends 패키지: import 시 기본 백엔드들을 레지스트리에 등록한다.

새 모델을 추가하려면 이 파일에 register() 한 줄을 추가하면 된다.
같은 클래스를 다른 파라미터로 여러 이름에 등록할 수 있다.
"""

from .base import (
    BackendNotConfigured,
    ClipGenerationError,
    ClipResult,
    ClipSpec,
    VideoBackend,
    available_models,
    backend_info,
    get_backend,
    register,
    unregister,
)
from .ltx import LtxBackend
from .sora import SoraBackend
from .veo import VeoBackend

# ---------------------------------------------------------------------- #
# 기본 백엔드 등록
# ---------------------------------------------------------------------- #
register("sora-2", SoraBackend, model="sora-2")
register("sora-2-pro", SoraBackend, model="sora-2-pro")
register("veo-3.1", VeoBackend)                      # settings.veo_model_default 사용
register("veo-3.1-fast", VeoBackend, model="veo-3.1-fast-generate-preview")
register("ltx-2.3", LtxBackend)                      # settings.ltx_endpoint_default 사용
register(
    "ltx-2.3-fast",
    LtxBackend,
    endpoint="fal-ai/ltx-2.3/text-to-video/fast",
    # Fast 변형은 6~20초(짝수) 지원
    supported_durations=(6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
)

__all__ = [
    "VideoBackend",
    "ClipSpec",
    "ClipResult",
    "ClipGenerationError",
    "BackendNotConfigured",
    "register",
    "unregister",
    "get_backend",
    "backend_info",
    "available_models",
]
