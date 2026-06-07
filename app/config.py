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

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """multi_modal_agent 설정값. .env 또는 환경 변수에서 읽는다."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # 보안
    # ------------------------------------------------------------------ #
    # 프론트엔드가 X-App-Key 헤더로 전달해야 하는 앱 키 목록(콤마 구분).
    # 예: APP_KEYS=key-for-web,key-for-mobile
    app_keys: str = ""

    # CORS 허용 오리진(콤마 구분). 운영에서는 프론트 도메인으로 제한할 것.
    cors_origins: str = "*"

    # ------------------------------------------------------------------ #
    # 외부 API 키 (해당 백엔드를 쓸 때만 필요)
    # ------------------------------------------------------------------ #
    openai_api_key: str = ""      # Sora 2 + LLM(파싱/스토리보드) + TTS(내레이션)
    gemini_api_key: str = ""      # Veo 3.1 (Google Gemini API)
    fal_api_key: str = ""         # LTX-2 (fal.ai 호스팅)

    # ------------------------------------------------------------------ #
    # LLM (PDF 파싱 / 스토리보드 변환)
    # ------------------------------------------------------------------ #
    openai_llm_model: str = "gpt-4o"
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

    # ------------------------------------------------------------------ #
    # 내레이션(TTS) — 선택 기능
    # ------------------------------------------------------------------ #
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"

    # ------------------------------------------------------------------ #
    # 저장소
    # ------------------------------------------------------------------ #
    data_dir: str = "data"             # 잡/클립/최종본 저장 루트

    # ------------------------------------------------------------------ #
    # 파생 헬퍼
    # ------------------------------------------------------------------ #
    @property
    def app_key_list(self) -> list[str]:
        return [k.strip() for k in self.app_keys.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def jobs_dir(self) -> Path:
        return Path(self.data_dir) / "jobs"

    def validate_runtime(self) -> None:
        """서버 기동 시 필수 설정 검증. 문제가 있으면 RuntimeError."""
        if not self.app_key_list:
            raise RuntimeError(
                "APP_KEYS 가 설정되지 않았습니다. .env 에 최소 1개의 앱 키를 "
                "설정해야 서버가 기동됩니다. 예) APP_KEYS=dev-key-1234"
            )
