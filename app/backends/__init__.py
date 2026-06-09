"""
backends 패키지: import 시 기본 백엔드들을 레지스트리에 등록한다.

새 모델을 추가하려면 이 파일에 register() 한 줄을 추가하면 된다.
같은 클래스를 다른 파라미터로 여러 이름에 등록할 수 있다.
"""

from .base import (
    DEFAULT_TEXT_EXPOSURE,
    TEXT_EXPOSURE_LEVELS,
    BackendNotConfigured,
    ClipGenerationError,
    ClipResult,
    ClipSpec,
    VideoBackend,
    apply_text_policy,
    available_models,
    backend_info,
    get_backend,
    negative_for,
    negative_prompt_for,
    normalize_text_exposure,
    register,
    text_clause_for,
    unregister,
)
from .ltx import LtxBackend
from .sora import SoraBackend
from .veo import VeoBackend
from .wan import WanBackend

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
    i2v_endpoint="fal-ai/ltx-2.3/image-to-video/fast",
    # Fast 변형은 6~20초(짝수) 지원
    supported_durations=(6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
)

# --- Alibaba Wan (Model Studio / DashScope) ---------------------------- #
# wan-2.2: 무음, 5초 고정, 480P/1080P (settings 기본 모델 ID 사용)
register(
    "wan-2.2",
    WanBackend,
    supported_durations=(5.0,),
    resolutions=("480p", "1080p"),
    supports_audio=False,   # wan-2.2 계열은 무음(네이티브 오디오 없음)
)
# wan-2.5: 오디오 생성(보이스오버 발화), 5/10초, 480/720/1080P
register(
    "wan-2.5",
    WanBackend,
    t2v_model="wan2.5-t2v-preview",
    i2v_model="wan2.5-i2v-preview",
    supported_durations=(5.0, 10.0),
    resolutions=("480p", "720p", "1080p"),
)
# wan-2.6: 2~15초(가변), 네이티브 오디오, 720P/1080P, 멀티샷 지원.
register(
    "wan-2.6",
    WanBackend,
    t2v_model="wan2.6-t2v",
    i2v_model="wan2.6-i2v",
    supported_durations=(4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0),
    resolutions=("720p", "1080p"),
)
# wan-2.7: 최신 세대. 2~15초(가변), 네이티브 오디오, 720P/1080P, 멀티샷·first/last 제어.
register(
    "wan-2.7",
    WanBackend,
    t2v_model="wan2.7-t2v",
    i2v_model="wan2.7-i2v",
    supported_durations=(4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0),
    resolutions=("720p", "1080p"),
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
    "normalize_text_exposure",
    "apply_text_policy",
    "negative_for",
    "negative_prompt_for",
    "text_clause_for",
    "TEXT_EXPOSURE_LEVELS",
    "DEFAULT_TEXT_EXPOSURE",
]
