"""
ads/manager.py
--------------
광고 파이프라인 잡 저장소 + 단계 게이팅 + 백그라운드 실행.

[설계 노트]
- 기존 JobManager(app/jobs)와 분리: 잡 모델/단계 구조가 다르고,
  "기존 API 와 독립"이라는 요구사항을 코드 수준에서도 지킨다.
- 단계 시작(begin_stage)은 단일 이벤트 루프에서 await 없이 수행되어
  검사-선점(check-and-set)이 원자적이다(중복 실행/선행 조건 위반 차단).
- 잡 상태는 메모리에 들고 변경 시마다 data/ad_jobs/{id}/job.json 으로
  영속화한다. 서버 재시작 시 in_progress 단계는 failed 로 마감한다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

from fastapi import HTTPException

from .schemas import AdJob, STAGE_NAMES, STAGE_REQUIRES

logger = logging.getLogger(__name__)


class AdJobManager:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, AdJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._load_existing()

    # ------------------------------------------------------------------ #
    # 조회/생성
    # ------------------------------------------------------------------ #
    def create(
        self, prompt: str, options,
        requested_by: str | None = None, requested_by_id: str | None = None,
    ) -> AdJob:
        job_id = uuid.uuid4().hex[:12]
        job = AdJob(
            id=job_id, prompt=prompt, options=options,
            requested_by=requested_by, requested_by_id=requested_by_id,
        )
        self._jobs[job_id] = job
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.persist(job)
        return job

    def get(self, job_id: str) -> Optional[AdJob]:
        return self._jobs.get(job_id)

    def get_or_404(self, job_id: str) -> AdJob:
        job = self.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail=f"잡을 찾을 수 없습니다: {job_id}"
            )
        return job

    def list(self) -> list[AdJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ------------------------------------------------------------------ #
    # 디렉토리
    # ------------------------------------------------------------------ #
    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def images_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "images"

    def videos_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "videos"

    # ------------------------------------------------------------------ #
    # 상태 갱신/영속화
    # ------------------------------------------------------------------ #
    def update(self, job: AdJob, **fields) -> None:
        for key, value in fields.items():
            setattr(job, key, value)
        job.touch()
        self.persist(job)

    def persist(self, job: AdJob) -> None:
        path = self.job_dir(job.id) / "job.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 - 영속화 실패가 서비스를 죽이면 안 됨
            logger.exception("광고 잡 영속화 실패: %s", job.id)

    # ------------------------------------------------------------------ #
    # 단계 게이팅 (★ 요구사항 3: videos 는 images 완료 후에만)
    # ------------------------------------------------------------------ #
    def begin_stage(self, job: AdJob, stage_name: str, force: bool = False) -> None:
        """
        단계 시작을 원자적으로 선점한다. 위반 시 HTTPException:
          412 — 선행 단계 미완료
          409 — 이미 실행 중이거나, 완료된 단계 재실행(force=False)
        이 메서드는 await 없이 동작하므로 단일 이벤트 루프에서
        레이스 컨디션이 발생하지 않는다.
        """
        if stage_name not in STAGE_NAMES:
            raise HTTPException(status_code=400, detail=f"알 수 없는 단계: {stage_name}")

        # 1) 선행 단계 검사
        for required in STAGE_REQUIRES[stage_name]:
            req = job.stage(required)
            if req.status != "completed":
                raise HTTPException(
                    status_code=412,
                    detail=(
                        f"'{stage_name}' 단계는 '{required}' 단계가 완료된 후에만 "
                        f"실행할 수 있습니다. (현재 '{required}' 상태: {req.status})"
                    ),
                )

        # 2) 자기 자신 상태 검사
        st = job.stage(stage_name)
        if st.status == "in_progress":
            raise HTTPException(
                status_code=409,
                detail=f"'{stage_name}' 단계가 이미 실행 중입니다.",
            )
        if st.status == "completed" and not force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{stage_name}' 단계가 이미 완료되었습니다. "
                    f"다시 실행하려면 ?force=true 를 사용하세요."
                ),
            )

        # 3) 후행 단계가 실행 중이면 산출물 정합성이 깨지므로 차단
        for later, requires in STAGE_REQUIRES.items():
            if stage_name in requires and job.stage(later).status == "in_progress":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'{later}' 단계가 실행 중이라 '{stage_name}' 단계를 "
                        f"다시 시작할 수 없습니다."
                    ),
                )

        # 4) 선점: in_progress 전환 + 이전 산출물 초기화
        st.begin()
        if stage_name == "storyboard":
            job.storyboard = None
        elif stage_name == "images":
            job.images = []
        elif stage_name == "videos":
            job.videos = []
            job.final_video_path = None
        elif stage_name == "pdf":
            job.pdf_path = None
        job.touch()
        self.persist(job)

    def finish_stage(self, job: AdJob, stage_name: str,
                     error: Optional[str] = None) -> None:
        st = job.stage(stage_name)
        if error is None:
            st.complete()
        else:
            st.fail(error)
        job.touch()
        self.persist(job)

    # ------------------------------------------------------------------ #
    # 백그라운드 실행
    # ------------------------------------------------------------------ #
    def start(self, job: AdJob, stage_name: str,
              runner: Callable[[], Awaitable[None]]) -> None:
        """단계 코루틴을 asyncio Task 로 실행한다(최종 방어선 포함)."""

        task_key = f"{job.id}:{stage_name}"

        async def _wrapped() -> None:
            try:
                await runner()
            except Exception as exc:  # noqa: BLE001 - 파이프라인 최종 방어선
                logger.exception("광고 잡 단계 미처리 예외: %s/%s", job.id, stage_name)
                self.finish_stage(job, stage_name, error=f"내부 오류: {exc}")
            finally:
                self._tasks.pop(task_key, None)

        task = asyncio.create_task(_wrapped(), name=f"ad-{task_key}")
        self._tasks[task_key] = task

    # ------------------------------------------------------------------ #
    # 재시작 복구
    # ------------------------------------------------------------------ #
    def _load_existing(self) -> None:
        for job_file in self.jobs_dir.glob("*/job.json"):
            try:
                job = AdJob.model_validate_json(job_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                logger.warning("광고 잡 파일 파싱 실패(무시): %s", job_file)
                continue
            for name in STAGE_NAMES:
                st = job.stage(name)
                if st.status == "in_progress":
                    st.fail("서버 재시작으로 작업이 중단되었습니다.")
            self._jobs[job.id] = job
            self.persist(job)
