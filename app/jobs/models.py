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
    request: dict = Field(default_factory=dict)   # 원 요청 기록(감사/디버깅)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()
