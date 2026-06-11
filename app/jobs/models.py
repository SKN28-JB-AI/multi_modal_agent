"""
jobs/models.py
--------------
잡(비동기 동영상 생성 작업) 상태 모델.

상태 흐름:
  queued → parsing(PDF만) → storyboarding(PDF만) → generating
         → postprocessing → completed | failed
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..schemas import Storyboard


class JobStatus(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"                 # ① PDF 파싱/이해
    STORYBOARDING = "storyboarding"     # ② 스토리보드 변환
    GENERATING = "generating"           # ③ 씬별 영상 생성
    POSTPROCESSING = "postprocessing"   # ④ 후처리
    COMPLETED = "completed"
    FAILED = "failed"


class SceneState(BaseModel):
    index: int
    status: str = "pending"             # pending / generating / completed / failed
    clip_path: Optional[str] = None
    error: Optional[str] = None
    # 백엔드 측 비디오 ID (Sora video_id 등). remix 시 원본 참조로 사용.
    backend_job_id: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job(BaseModel):
    id: str
    mode: str                           # "message" | "pdf"
    model: str                          # 비디오 백엔드 이름
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    error: Optional[str] = None
    storyboard: Optional[Storyboard] = None
    scenes: list[SceneState] = Field(default_factory=list)
    final_path: Optional[str] = None
    subtitles_path: Optional[str] = None
    # 스토리보드 씬별 실제(또는 스케일된) 길이 — SRT/remix 재계산용
    scene_durations: Optional[list[float]] = None
    request: dict = Field(default_factory=dict)   # 원 요청 기록(감사/디버깅)
    # 요청자(감사/표시용). JWT 인증 시 기록되며, 앱 키 호출 등
    # 사용자 정보가 없으면 None 으로 남는다(프론트 미표시).
    requested_by: Optional[str] = None      # 표시 이름(username)
    requested_by_id: Optional[str] = None   # auth-server user id(sub)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    # 처리 시작/종료 시각 — 상태 전이 시 JobManager 가 기록
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def touch(self) -> None:
        self.updated_at = _now()
