"""
routers/jobs.py
---------------
잡 상태 조회 / 결과물 다운로드.

  GET /v1/jobs            : 잡 목록
  GET /v1/jobs/{id}       : 잡 상태(스토리보드/씬별 진행 포함)
  GET /v1/jobs/{id}/video : 완성된 MP4 다운로드
  GET /v1/jobs/{id}/subtitles : SRT 자막(있는 경우)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..jobs import Job, JobStatus
from ..schemas import JobStatusResponse, SceneStateOut
from ..security import require_app_key

router = APIRouter(
    prefix="/v1/jobs", tags=["jobs"], dependencies=[Depends(require_app_key)]
)


def _to_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        mode=job.mode,
        model=job.model,
        status=job.status.value,
        progress=round(job.progress, 3),
        error=job.error,
        storyboard=job.storyboard,
        scenes=[
            SceneStateOut(index=s.index, status=s.status, error=s.error)
            for s in job.scenes
        ],
        video_url=(
            f"/v1/jobs/{job.id}/video"
            if job.status == JobStatus.COMPLETED and job.final_path else None
        ),
        subtitles_url=(
            f"/v1/jobs/{job.id}/subtitles" if job.subtitles_path else None
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _get_job_or_404(request: Request, job_id: str) -> Job:
    job = request.app.state.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없습니다.")
    return job


@router.get("", response_model=list[JobStatusResponse])
async def list_jobs(request: Request):
    return [_to_response(j) for j in request.app.state.job_manager.list()]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(request: Request, job_id: str):
    return _to_response(_get_job_or_404(request, job_id))


@router.get("/{job_id}/video")
async def download_video(request: Request, job_id: str):
    job = _get_job_or_404(request, job_id)
    if job.status != JobStatus.COMPLETED or not job.final_path:
        raise HTTPException(
            status_code=409,
            detail=f"아직 완료되지 않은 잡입니다(status={job.status.value}).",
        )
    path = Path(job.final_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="결과 파일이 삭제되었습니다.")
    return FileResponse(
        path, media_type="video/mp4", filename=f"{job_id}.mp4"
    )


@router.get("/{job_id}/subtitles")
async def download_subtitles(request: Request, job_id: str):
    job = _get_job_or_404(request, job_id)
    if not job.subtitles_path or not Path(job.subtitles_path).exists():
        raise HTTPException(status_code=404, detail="자막 파일이 없습니다.")
    return FileResponse(
        Path(job.subtitles_path),
        media_type="application/x-subrip", filename=f"{job_id}.srt",
    )
