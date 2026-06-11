"""
ads/schemas.py
--------------
광고 파이프라인(스토리보드 → 이미지 → 비디오 → 기획서) 데이터 모델.

스토리보드 스키마는 사내에서 검증된 광고 콘티 JSON 형식
(project / concept / mood / music / cuts[...] / cta / logo)을 따르며,
OpenAI 구조화 출력(response_format)에 그대로 사용된다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ====================================================================== #
# 스토리보드 (1단계 산출물 / LLM 구조화 출력 스키마)
# ====================================================================== #
class Music(BaseModel):
    """배경 음악 연출 정보."""

    genre: str = Field(description="음악 장르 (예: 힙합 / 일렉트로닉 비트)")
    bpm: int = Field(description="템포 BPM (예: 120)")
    key_moment: str = Field(description="음악적 키 모먼트 설명")


class Cut(BaseModel):
    """광고 비디오의 단일 컷(장면) 정의."""

    cut: int = Field(description="컷 번호 (1부터 시작)")
    timecode: str = Field(description="타임코드 'MM:SS-MM:SS' 형식")
    duration_sec: int = Field(description="컷 길이(초)")
    title: str = Field(description="컷 제목")
    scene: str = Field(description="장면 개요(장소·인물·행동)")
    visual: str = Field(description="화면 연출 상세 묘사")
    camera: str = Field(description="카메라 워크")
    on_screen_text: str = Field(default="", description="화면 자막(없으면 빈 문자열)")
    voiceover: str = Field(default="", description="성우 대사(없으면 빈 문자열)")
    sfx: str = Field(default="", description="효과음/사운드 묘사")
    transition: str = Field(default="", description="다음 컷으로의 전환 기법")


class AdStoryboard(BaseModel):
    """광고 비디오 스토리보드 전체."""

    project: str = Field(description="프로젝트/캠페인 이름")
    concept: str = Field(description="광고 핵심 컨셉 한 줄 설명")
    target: str = Field(description="타겟 고객층")
    mood: list[str] = Field(description="무드 키워드 목록")
    total_duration_sec: int = Field(description="전체 길이(초)")
    aspect_ratio: str = Field(default="16:9", description="화면 비율")
    format: str = Field(default="", description="배포 포맷 (예: 유튜브 인스트림)")
    music: Music
    cuts: list[Cut]
    cta: str = Field(default="", description="행동 유도 문구")
    logo: str = Field(default="", description="브랜드/로고 표기")


# ====================================================================== #
# 요청
# ====================================================================== #
AspectRatio = Literal["16:9", "9:16"]
Resolution = Literal["720p", "1080p"]


class AdStoryboardOptions(BaseModel):
    """1단계 스토리보드 생성 옵션(모두 선택)."""

    cut_count: int = Field(default=3, ge=1, le=8, description="컷 수")
    total_duration_sec: int = Field(default=16, ge=4, le=60, description="전체 길이(초)")
    aspect_ratio: AspectRatio = "16:9"
    resolution: Resolution = "1080p"
    locale: str = Field(default="ko-KR", description="콘텐츠 언어")
    brand: str = Field(default="JB금융그룹", description="브랜드/로고 표기")


class AdStoryboardRequest(BaseModel):
    """1단계: 스토리보드 생성 요청."""

    prompt: str = Field(min_length=1, max_length=8000, description="광고 비디오 프롬프트")
    options: AdStoryboardOptions = Field(default_factory=AdStoryboardOptions)

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("프롬프트가 비어 있습니다.")
        return v


class AdImagesRequest(BaseModel):
    """2단계: 컷별 첫 장면 이미지 생성 요청."""

    model: Optional[str] = Field(
        default=None,
        description="이미지 모델 이름 (GET /v2/ads/image-models 참고). "
                    "미지정 시 서버 기본값(gpt-image-2).",
    )


class AdVideosRequest(BaseModel):
    """3단계: 이미지 기반 컷 비디오 생성 요청."""

    model: Optional[str] = Field(
        default=None,
        description="비디오 백엔드 이름 (GET /v1/models 참고). "
                    "미지정 시 서버 기본값.",
    )
    text_exposure: Optional[Literal["none", "minimal", "moderate", "full"]] = Field(
        default=None,
        description=(
            "화면 글자 노출 단계: none/minimal/moderate/full. 미지정 시 서버 "
            "기본값(TEXT_EXPOSURE_DEFAULT, 기본 minimal)."
        ),
    )
    logo_outro: Optional[bool] = Field(
        default=None,
        description=(
            "광고 마지막에 로고 엔드카드(아웃트로)를 붙일지 여부. 미지정 시 "
            "서버 기본값(LOGO_OUTRO_ENABLED, 기본 false). 배경색은 LLM 추천."
        ),
    )


# ====================================================================== #
# 잡(Job) 상태
# ====================================================================== #
StageStatus = Literal["pending", "in_progress", "completed", "failed"]

STAGE_NAMES = ("storyboard", "images", "videos", "pdf")

# 단계별 선행 조건. ★ videos 는 images 완료가 필수(요구사항 3).
#   pdf 는 storyboard 만 요구 — 2·3단계와 무관하게 실행 가능(요구사항 4).
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "storyboard": (),
    "images": ("storyboard",),
    "videos": ("storyboard", "images"),
    "pdf": ("storyboard",),
}


class StageState(BaseModel):
    """파이프라인 한 단계의 진행 상태."""

    status: StageStatus = "pending"
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def begin(self) -> None:
        self.status = "in_progress"
        self.error = None
        self.started_at = _now_iso()
        self.finished_at = None

    def complete(self) -> None:
        self.status = "completed"
        self.error = None
        self.finished_at = _now_iso()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.finished_at = _now_iso()


class CutAsset(BaseModel):
    """컷 하나에 대응하는 산출물(이미지 또는 비디오 클립)."""

    cut: int
    status: StageStatus = "pending"
    path: Optional[str] = None
    error: Optional[str] = None
    # 비디오 전용: 백엔드 측 잡 ID(Sora video_id 등)
    backend_job_id: Optional[str] = None
    # 생성에 실제 사용된 프롬프트(디버깅/재현용)
    prompt_used: Optional[str] = None


class AdJob(BaseModel):
    """광고 파이프라인 잡 전체 상태(data/ad_jobs/{id}/job.json 으로 영속화)."""

    id: str
    prompt: str
    options: AdStoryboardOptions = Field(default_factory=AdStoryboardOptions)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    # 단계별 상태
    storyboard_stage: StageState = Field(default_factory=StageState)
    images_stage: StageState = Field(default_factory=StageState)
    videos_stage: StageState = Field(default_factory=StageState)
    pdf_stage: StageState = Field(default_factory=StageState)

    # 단계별 사용 모델(감사/재현용)
    image_model: Optional[str] = None
    video_model: Optional[str] = None

    # 산출물
    storyboard: Optional[AdStoryboard] = None
    images: list[CutAsset] = Field(default_factory=list)
    videos: list[CutAsset] = Field(default_factory=list)
    final_video_path: Optional[str] = None
    pdf_path: Optional[str] = None

    def stage(self, name: str) -> StageState:
        mapping = {
            "storyboard": self.storyboard_stage,
            "images": self.images_stage,
            "videos": self.videos_stage,
            "pdf": self.pdf_stage,
        }
        if name not in mapping:
            raise KeyError(f"알 수 없는 단계: {name}")
        return mapping[name]

    def touch(self) -> None:
        self.updated_at = _now_iso()


# ====================================================================== #
# 응답
# ====================================================================== #
class AdJobAccepted(BaseModel):
    """비동기 단계 접수 응답(202)."""

    job_id: str
    stage: str
    status: StageStatus
    status_url: str
    message: str


class StageStateOut(BaseModel):
    status: StageStatus
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: Optional[float] = None  # 단계 소요시간(초, 종료 시)


class CutAssetOut(BaseModel):
    cut: int
    status: StageStatus
    error: Optional[str] = None
    url: Optional[str] = None


class AdJobStatusResponse(BaseModel):
    """GET /v2/ads/{job_id} 응답."""

    job_id: str
    prompt: str
    options: AdStoryboardOptions
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    stages: dict[str, StageStateOut]
    storyboard: Optional[AdStoryboard] = None
    images: list[CutAssetOut] = Field(default_factory=list)
    videos: list[CutAssetOut] = Field(default_factory=list)
    storyboard_url: Optional[str] = None
    final_video_url: Optional[str] = None
    pdf_url: Optional[str] = None
    created_at: str
    updated_at: str


class ImageModelInfo(BaseModel):
    name: str
    provider: str
    description: str
    configured: bool
    default: bool = False


class ImageModelsResponse(BaseModel):
    models: list[ImageModelInfo]
