"""
v1 잡 단계 타임스탬프(started_at/finished_at) + 소요시간 계산 테스트.
"""
import tempfile
import time
from pathlib import Path

from app.jobs.manager import JobManager
from app.jobs.models import Job, JobStatus
from app.timeutil import iso_duration_sec


def _manager(tmp: str) -> JobManager:
    return JobManager(Path(tmp))


def test_started_and_finished_recorded_on_transitions():
    with tempfile.TemporaryDirectory() as d:
        m = _manager(d)
        job = m.create(mode="message", model="veo-3.1", request={})
        assert job.started_at is None and job.finished_at is None

        m.update(job, status=JobStatus.GENERATING, progress=0.2)
        assert job.started_at is not None
        assert job.finished_at is None
        first = job.started_at

        time.sleep(0.02)
        m.update(job, status=JobStatus.POSTPROCESSING, progress=0.9)
        assert job.started_at == first  # 최초 1회만 기록

        m.update(job, status=JobStatus.COMPLETED, progress=1.0)
        assert job.finished_at is not None
        dur = iso_duration_sec(job.started_at, job.finished_at)
        assert dur is not None and dur >= 0


def test_failed_job_gets_finished_at():
    with tempfile.TemporaryDirectory() as d:
        m = _manager(d)
        job = m.create(mode="pdf", model="sora-2", request={})
        m.update(job, status=JobStatus.PARSING)
        m.update(job, status=JobStatus.FAILED, error="boom")
        assert job.started_at and job.finished_at


def test_restart_recovery_sets_finished_at():
    with tempfile.TemporaryDirectory() as d:
        m = _manager(d)
        job = m.create(mode="message", model="wan", request={})
        m.update(job, status=JobStatus.GENERATING)

        m2 = _manager(d)  # 재시작 시뮬레이션
        recovered = m2.get(job.id)
        assert recovered.status == JobStatus.FAILED
        assert recovered.finished_at


def test_old_job_json_backcompat():
    # 필드가 없던 기존 job.json 도 로드 가능해야 한다
    j = Job.model_validate({"id": "x", "mode": "message", "model": "wan", "request": {}})
    assert j.started_at is None and j.finished_at is None


def test_iso_duration_sec_edge_cases():
    assert iso_duration_sec(None, None) is None
    assert iso_duration_sec("2026-06-10T00:00:00+00:00", None) is None
    assert iso_duration_sec("not-a-date", "2026-06-10T00:00:01+00:00") is None
    assert iso_duration_sec(
        "2026-06-10T00:00:00+00:00", "2026-06-10T00:01:30+00:00"
    ) == 90.0
