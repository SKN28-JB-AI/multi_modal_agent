"""
jobs/manager.py
---------------
잡 저장소 + 백그라운드 실행 관리.

[설계 노트]
- 동영상 생성은 수 분 단위라 요청-응답 안에서 처리할 수 없다.
  POST 는 즉시 job_id 를 반환하고, 파이프라인은 asyncio Task 로 돈다.
- 잡 상태는 메모리에 들고, 변경 시마다 data/jobs/{id}/job.json 으로
  영속화한다(서버 재시작 시 조회는 가능, 실행 중이던 잡은 failed 처리).
- 단일 프로세스 가정. 다중 워커로 확장하려면 저장소를 Redis/DB 로,
  실행을 Celery/Arq 등으로 교체한다(이 모듈만 바꾸면 됨).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .models import Job, JobStatus, _now

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._load_existing()

    # ------------------------------------------------------------------ #
    # 조회/생성
    # ------------------------------------------------------------------ #
    def create(self, mode: str, model: str, request: dict) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, mode=mode, model=model, request=request)
        self._jobs[job_id] = job
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    # ------------------------------------------------------------------ #
    # 상태 갱신
    # ------------------------------------------------------------------ #
    def update(self, job: Job, **fields) -> None:
        """잡 필드를 갱신하고 디스크에 영속화한다."""
        for key, value in fields.items():
            setattr(job, key, value)
        if "status" in fields:
            self._track_times(job, fields["status"])
        job.touch()
        self.persist(job)

    @staticmethod
    def _track_times(job: Job, new_status: JobStatus | str) -> None:
        """상태 전이 시 처리 시작/종료 시각을 기록한다."""
        status = JobStatus(new_status)
        if job.started_at is None and status != JobStatus.QUEUED:
            job.started_at = _now()
        if (status in (JobStatus.COMPLETED, JobStatus.FAILED)
                and job.finished_at is None):
            job.finished_at = _now()

    def persist(self, job: Job) -> None:
        path = self.job_dir(job.id) / "job.json"
        try:
            path.write_text(
                job.model_dump_json(indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 - 영속화 실패가 서비스를 죽이면 안 됨
            logger.exception("잡 영속화 실패: %s", job.id)

    # ------------------------------------------------------------------ #
    # 백그라운드 실행
    # ------------------------------------------------------------------ #
    def start(self, job: Job, runner: Callable[[], Awaitable[None]]) -> None:
        """파이프라인 코루틴을 asyncio Task 로 실행한다."""

        async def _wrapped() -> None:
            try:
                await runner()
            except Exception as exc:  # noqa: BLE001 - 파이프라인 최종 방어선
                logger.exception("잡 실행 중 미처리 예외: %s", job.id)
                self.update(
                    job, status=JobStatus.FAILED,
                    error=f"내부 오류: {exc}",
                )
            finally:
                self._tasks.pop(job.id, None)

        task = asyncio.create_task(_wrapped(), name=f"job-{job.id}")
        self._tasks[job.id] = task

    # ------------------------------------------------------------------ #
    # 재시작 복구
    # ------------------------------------------------------------------ #
    def _load_existing(self) -> None:
        """디스크의 기존 잡을 읽는다. 실행 중이던 잡은 failed 로 마감."""
        for job_file in self.jobs_dir.glob("*/job.json"):
            try:
                job = Job.model_validate_json(
                    job_file.read_text(encoding="utf-8")
                )
            except Exception:  # noqa: BLE001
                logger.warning("잡 파일 파싱 실패(무시): %s", job_file)
                continue
            if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.status = JobStatus.FAILED
                job.error = "서버 재시작으로 작업이 중단되었습니다."
                # 마지막 갱신 시각을 종료 시각으로 간주
                job.finished_at = job.finished_at or job.updated_at
                job.touch()
            self._jobs[job.id] = job
            self.persist(job)
