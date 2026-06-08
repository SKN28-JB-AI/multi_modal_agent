"""
schemas.py
----------
API 요청/응답 및 스토리보드 데이터 모델 (Pydantic v2).

[스토리보드]
PDF 기획서 모드의 ② 단계 산출물. LLM 이 이 스키마에 맞는 JSON 을
생성하고, 파이프라인은 scenes 를 순서대로 비디오 백엔드에 넘긴다.
- scene.prompt        : 영상 생성 프롬프트(영어 권장 — 모델 성능이 가장 좋음)
- scene.on_screen_text: 화면 카피/자막(원문 언어). 생성 모델은 텍스트
                        렌더링이 약하므로 영상에 넣지 않고 SRT 로 뽑는다.
- narration_script    : 내레이션 대본(옵션 TTS 합성에 사용)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

AspectRatio = Literal["16:9", "9:16"]
Resolution = Literal["720p", "1080p"]


# ---------------------------------------------------------------------- #
# 스토리보드
# ---------------------------------------------------------------------- #
class Scene(BaseModel):
    index: int = Field(ge=0)
    prompt: str = Field(min_length=1, description="영상 생성 프롬프트(영어 권장)")
    duration_sec: float = Field(default=6.0, ge=2.0, le=20.0)
    audio_description: str = Field(default="", description="효과음/배경음 묘사")
    narration: str = Field(
        default="",
        description=(
            "이 씬에서 발화될 내레이션/대사 한 문장(원문 언어). "
            "비디오 모델이 보이스오버로 직접 발화한다."
        ),
    )
    on_screen_text: str = Field(default="", description="화면 카피(후처리 자막용)")


class Storyboard(BaseModel):
    title: str = ""
    summary: str = ""
    target_audience: str = ""
    narration_script: str = Field(default="", description="내레이션 대본(전체)")
    scenes: list[Scene] = Field(min_length=1, max_length=12)


# ---------------------------------------------------------------------- #
# 요청
# ---------------------------------------------------------------------- #
class GenerationOptions(BaseModel):
    """두 모드 공통의 영상 생성 옵션."""

    aspect_ratio: AspectRatio = "16:9"
    resolution: Resolution = "1080p"
    generate_audio: bool = True


class MessageRequest(GenerationOptions):
    """메시지(단일 프롬프트) 모드 요청."""

    prompt: str = Field(min_length=1, max_length=4000)
    model: str = Field(description="사용할 비디오 백엔드 이름 (GET /v1/models 참고)")
    duration_sec: Optional[float] = Field(
        default=None, ge=2.0, le=20.0,
        description="클립 길이(초). 백엔드 지원 값으로 자동 보정됨",
    )
    enhance_prompt: Optional[bool] = Field(
        default=None,
        description=(
            "비디오 생성 전 OpenAI 기본 모델로 프롬프트를 비디오 모델 맞춤형으로 "
            "변환할지 여부. 미지정 시 서버 기본값(ENHANCE_MESSAGE_PROMPT)을 따른다. "
            "OpenAI 키가 없거나 변환 실패 시 원본 프롬프트로 자동 폴백한다."
        ),
    )


class PdfJobOptions(GenerationOptions):
    """PDF 기획서 모드 옵션 (multipart 의 options 필드, JSON 문자열)."""

    generation_mode: Literal["single", "scenes"] = Field(
        default="single",
        description=(
            "single: 스토리보드를 샷 타임라인 프롬프트로 합성해 한 번의 "
            "생성 요청으로 영상 1개 생성(백엔드 최대 길이로 보정됨). "
            "scenes: 씬별 클립 생성 후 FFmpeg 결합(긴 광고용)."
        ),
    )
    target_total_duration_sec: float = Field(default=24.0, ge=4.0, le=120.0)
    max_scenes: int = Field(default=4, ge=1, le=8)
    language: str = Field(default="ko", description="카피/내레이션 언어")
    enable_narration: bool = Field(
        default=False, description="OpenAI TTS 로 내레이션 합성 (OPENAI_API_KEY 필요)"
    )
    burn_subtitles: bool = Field(
        default=False, description="SRT 자막을 영상에 굽기(재인코딩 발생)"
    )
    logo_name: Optional[str] = Field(
        default=None,
        description=(
            "서버 logos/ 폴더의 로고 파일명 (GET /v1/logos 로 조회). "
            "미지정 시 default.png 또는 첫 파일이 자동 적용된다."
        ),
    )


class RemixRequest(BaseModel):
    """완료된 잡의 특정 씬을 프롬프트로 부분 수정(remix)하는 요청."""

    prompt: str = Field(min_length=1, max_length=4000,
                        description="수정 지시 프롬프트")
    scene_index: int = Field(default=0, ge=0,
                             description="수정할 씬 번호(메시지 모드는 0)")


# ---------------------------------------------------------------------- #
# 응답
# ---------------------------------------------------------------------- #
class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    detail: str = "잡이 생성되었습니다. GET /v1/jobs/{job_id} 로 상태를 조회하세요."


class SceneStateOut(BaseModel):
    index: int
    status: str
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    mode: str
    model: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    error: Optional[str] = None
    storyboard: Optional[Storyboard] = None
    scenes: list[SceneStateOut] = []
    video_url: Optional[str] = None     # 완료 시 다운로드 경로
    subtitles_url: Optional[str] = None # SRT 가 생성된 경우
    created_at: str
    updated_at: str


class BackendInfo(BaseModel):
    name: str
    provider: str
    description: str
    configured: bool          # 필요한 API 키가 설정되어 있는지
    supported_durations: list[float]
    supports_remix: bool = False
    supports_image_input: bool = False


class ModelsResponse(BaseModel):
    models: list[BackendInfo]
