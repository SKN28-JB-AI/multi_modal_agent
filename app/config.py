"""
config.py
---------
서비스 전역 설정. .env 파일과 환경 변수에서 로드한다.

[설계 노트]
- pydantic-settings 사용: 타입 검증 + .env 자동 로드.
- 비디오 백엔드별 API 키는 선택 사항이다. 키가 없는 백엔드는
  GET /v1/models 에서 configured=false 로 표시되고, 사용 시 503 을 반환한다.
- APP_KEYS 는 필수다. 비어 있으면 서버가 기동을 거부한다(보안 기본값).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """multi_modal_agent 설정값. .env 또는 환경 변수에서 읽는다."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        설정 소스 우선순위(높음→낮음): init > .env(dotenv) > 환경변수 > secrets.

        pydantic-settings 기본 순서는 env > dotenv 라, 컨테이너/OS 환경변수가
        .env 를 덮어쓴다. 본 서비스는 '소스의 .env 파일을 가장 우선'으로 사용하는
        요구사항이라 dotenv 를 환경변수 앞에 둔다. 단, 명시적 init 인자(테스트/
        프로그램적 주입)는 그대로 최상위를 유지한다.
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    # ------------------------------------------------------------------ #
    # 보안
    # ------------------------------------------------------------------ #
    # 프론트엔드가 X-App-Key 헤더로 전달해야 하는 앱 키 목록(콤마 구분).
    # 예: APP_KEYS=key-for-web,key-for-mobile
    app_keys: str = ""

    # CORS 허용 오리진(콤마 구분). 운영에서는 프론트 도메인으로 제한할 것.
    cors_origins: str = "*"

    # ------------------------------------------------------------------ #
    # auth-server(OAuth 2.1) JWT 자체검증 (X-App-Key 와 병행)
    # ------------------------------------------------------------------ #
    # 발급자(iss). 비어 있으면 JWT 검증 비활성(앱 키만 사용).
    # 참조 백엔드(examples/backend)와 동일하게 ISSUER 와 JWKS_URL 을 분리한다:
    # iss 는 외부 URL(예: http://localhost:9000), JWKS 는 컨테이너 내부 호스트
    # (예: http://auth-server:9000/jwks.json) 로 가져올 수 있다.
    auth_issuer: str = ""              # env: AUTH_ISSUER
    jwks_url: str = ""                 # env: JWKS_URL (비면 auth_issuer + /jwks.json)
    # 검증 대상(aud). 비어 있으면 aud 검증 생략(참조 백엔드와 동일).
    auth_audience: str = ""            # env: AUTH_AUDIENCE
    # 보호 API 호출에 필요한 스코프. 비우면 스코프 검증 생략.
    auth_required_scope: str = "api"   # env: AUTH_REQUIRED_SCOPE
    # exp/nbf 시계 오차 허용(초) + JWKS 캐시 TTL(초)
    auth_leeway_sec: int = 5           # env: AUTH_LEEWAY_SEC
    jwks_cache_lifespan_sec: int = 300 # env: JWKS_CACHE_LIFESPAN_SEC

    # ------------------------------------------------------------------ #
    # 외부 API 키 (해당 백엔드를 쓸 때만 필요)
    # ------------------------------------------------------------------ #
    openai_api_key: str = ""      # Sora 2 + LLM(파싱/스토리보드) + TTS(내레이션)
    gemini_api_key: str = ""      # Veo 3.1 (Google Gemini API)
    fal_api_key: str = ""         # LTX-2 (fal.ai 호스팅)
    dashscope_api_key: str = ""   # Alibaba Model Studio (Wan 비디오 + Qwen-Image)

    # Alibaba Model Studio(DashScope) 리전 엔드포인트 베이스.
    # International(싱가포르) 기본값. 중국 본토는 https://dashscope.aliyuncs.com/api/v1
    # 미국(버지니아)은 https://dashscope-us.aliyuncs.com/api/v1 로 덮어쓴다.
    # 주의: 모델·엔드포인트·API 키는 같은 리전이어야 한다(크로스 리전 호출 실패).
    dashscope_base_url: str = "https://dashscope-intl.aliyuncs.com/api/v1"

    # ------------------------------------------------------------------ #
    # LLM (PDF 파싱 / 스토리보드 변환)
    # ------------------------------------------------------------------ #
    openai_llm_model: str = "gpt-4o"
    # 메시지 모드(/v1/videos/message)에서 비디오 생성 전에 입력 프롬프트를
    # 비디오 모델 맞춤 프롬프트로 변환할지 여부. OpenAI 기본 모델 사용.
    # 요청 본문의 enhance_prompt 로 건당 덮어쓸 수 있다.
    enhance_message_prompt: bool = True

    # PDF 페이지를 비전 LLM에 넘길 때의 최대 페이지 수(비용 가드).
    pdf_max_pages: int = 10
    # PDF 페이지 렌더링 DPI (높을수록 선명하지만 토큰/전송량 증가).
    pdf_render_dpi: int = 110

    # ------------------------------------------------------------------ #
    # 비디오 백엔드 공통
    # ------------------------------------------------------------------ #
    poll_interval_sec: float = 10.0    # 외부 API 폴링 간격
    poll_timeout_sec: float = 1200.0   # 클립 1개당 최대 대기 시간
    clip_retries: int = 1              # 클립 생성 실패 시 재시도 횟수
    max_concurrent_clips: int = 1      # 씬 동시 생성 수(비용·레이트리밋 가드)

    # 백엔드별 모델 ID (API 측 변경에 대비해 .env 로 덮어쓰기 가능)
    sora_model_default: str = "sora-2"
    veo_model_default: str = "veo-3.1-generate-preview"
    veo_fast_model: str = "veo-3.1-fast-generate-preview"
    ltx_endpoint_default: str = "fal-ai/ltx-2.3/text-to-video"
    ltx_fast_endpoint: str = "fal-ai/ltx-2.3/text-to-video/fast"
    fal_queue_base: str = "https://queue.fal.run"

    # Alibaba Wan 비디오 모델 ID (API 측 변경에 대비해 .env 로 덮어쓰기 가능)
    # text-to-video 와 image-to-video 는 모델 ID 가 다르므로 쌍으로 관리한다.
    wan_t2v_model_default: str = "wan2.2-t2v-plus"   # 무음, 고정 5초, 480P/1080P
    wan_i2v_model_default: str = "wan2.2-i2v-plus"   # 무음, 고정 5초, 480P/1080P

    # ------------------------------------------------------------------ #
    # 광고 파이프라인 (/v2/ads — 스토리보드→이미지→비디오→기획서)
    # ------------------------------------------------------------------ #
    # 1단계 스토리보드 생성 LLM
    ad_storyboard_model: str = "gpt-4o"
    # 2단계 이미지 모델 기본값 (요청에서 model 로 덮어쓰기 가능)
    ad_image_model_default: str = "gpt-image-2"
    # 3단계 비디오 모델 기본값 (요청에서 model 로 덮어쓰기 가능)
    ad_video_model_default: str = "veo-3.1"
    # Google Imagen 모델 ID (Gemini API)
    imagen_model_default: str = "imagen-4.0-generate-001"
    # fal.ai 이미지 생성 엔드포인트 (FLUX)
    fal_image_endpoint_default: str = "fal-ai/flux/dev"
    # Alibaba Qwen-Image 텍스트→이미지 모델 ID (DashScope multimodal-generation)
    qwen_image_model_default: str = "qwen-image-2.0-pro"
    # LTX image-to-video 엔드포인트 (text-to-video 와 별도)
    ltx_i2v_endpoint_default: str = "fal-ai/ltx-2.3/image-to-video"
    ltx_i2v_fast_endpoint: str = "fal-ai/ltx-2.3/image-to-video/fast"
    # 이미지 1장당 최대 대기 시간(초) — 비디오보다 훨씬 짧다
    image_poll_timeout_sec: float = 300.0
    # 컷 이미지 생성 실패 시 재시도 횟수
    image_retries: int = 1
    # 기획서 PDF 한글 폰트 TTF 경로 (미지정 시 OS 폰트 자동 탐색)
    ad_pdf_font_path: str = ""
    # 광고 마지막 로고 아웃트로(엔드카드). 기본은 옵션(요청 logo_outro=true 시).
    logo_outro_enabled: bool = False
    logo_outro_duration_sec: float = 2.5
    logo_outro_fade_sec: float = 0.4
    logo_outro_scale_ratio: float = 0.42   # 로고 폭 = 프레임 폭 × 비율
    # 배경색 폴백(LLM 추천 실패/키 없음 시). JB 브랜드 네이비.
    logo_outro_bg_default: str = "#134A8E"
    # 화면 글자 노출 기본 단계: none / minimal / moderate / full.
    # 요청에서 text_exposure 로 건당 덮어쓸 수 있다.
    text_exposure_default: str = "minimal"
    # 자막 번인(burn_subtitles)에 쓸 한글 폰트 TTF/TTC 경로.
    # 미지정 시 OS 별 한글 폰트(맑은고딕/나눔/Noto CJK)를 자동 탐색한다.
    subtitle_font_path: str = ""

    # ------------------------------------------------------------------ #
    # 내레이션(TTS) — 선택 기능
    # ------------------------------------------------------------------ #
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"

    # ------------------------------------------------------------------ #
    # 저장소
    # ------------------------------------------------------------------ #
    data_dir: str = "data"             # 잡/클립/최종본 저장 루트
    # 브랜드 로고 보관 폴더. PDF 모드 요청 시 이 폴더의 로고가 적용된다.
    # 우선순위: 요청의 logo 업로드 > options.logo_name > default.png > 첫 파일.
    logos_dir: str = "logos"

    # ------------------------------------------------------------------ #
    # 로고 오버레이 스타일 (자연스러운 워터마크 처리)
    # ------------------------------------------------------------------ #
    # 로고 가로폭 = 영상 가로폭 × 비율. 0.12 = 12% (방송 워터마크 수준)
    logo_scale_ratio: float = 0.12
    # 불투명도 0~1. 1.0 은 원본 그대로, 0.8 전후가 자연스럽다.
    logo_opacity: float = 0.82
    # 위치: top-right / top-left / bottom-right / bottom-left
    logo_position: str = "top-right"
    # 가장자리 여백 = 영상 가로폭 × 비율
    logo_margin_ratio: float = 0.03
    # 페이드인 시간(초). 0 이면 즉시 표시.
    logo_fade_in_sec: float = 0.6

    # ------------------------------------------------------------------ #
    # 파생 헬퍼
    # ------------------------------------------------------------------ #
    @property
    def app_key_list(self) -> list[str]:
        return [k.strip() for k in self.app_keys.split(",") if k.strip()]

    @property
    def effective_jwks_url(self) -> str:
        """JWKS_URL 명시값 우선, 없으면 auth_issuer 로부터 유도."""
        if self.jwks_url.strip():
            return self.jwks_url.strip()
        if self.auth_issuer.strip():
            return self.auth_issuer.rstrip("/") + "/jwks.json"
        return ""

    @property
    def jwt_enabled(self) -> bool:
        """발급자와 JWKS 가 모두 설정되어야 JWT 검증을 켠다."""
        return bool(self.auth_issuer.strip() and self.effective_jwks_url)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def jobs_dir(self) -> Path:
        return Path(self.data_dir) / "jobs"

    @property
    def ad_jobs_dir(self) -> Path:
        return Path(self.data_dir) / "ad_jobs"

    def validate_runtime(self) -> None:
        """서버 기동 시 필수 설정 검증. 문제가 있으면 RuntimeError."""
        if not self.app_key_list:
            raise RuntimeError(
                "APP_KEYS 가 설정되지 않았습니다. .env 에 최소 1개의 앱 키를 "
                "설정해야 서버가 기동됩니다. 예) APP_KEYS=dev-key-1234"
            )
