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

from ..backends import BackendNotConfigured, get_backend
from ..jobs import Job, JobStatus
from ..schemas import (
    JobCreatedResponse, JobStatusResponse, RemixRequest, SceneStateOut,
)
from ..coupons import require_video_coupon
from ..security import require_auth
from ..timeutil import iso_duration_sec

router = APIRouter(
    prefix="/v1/jobs", tags=["jobs"], dependencies=[Depends(require_auth)]
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
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_sec=iso_duration_sec(job.started_at, job.finished_at),
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


@router.post("/{job_id}/remix", response_model=JobCreatedResponse,
             status_code=202,
             dependencies=[Depends(require_video_coupon)])  # 영상 쿠폰 1개 차감
async def remix_job(request: Request, job_id: str, body: RemixRequest):
    """
    완료된 잡의 특정 씬을 프롬프트로 부분 수정(remix)한다.
    새 잡(mode="remix")이 만들어지며, 수정된 씬 + 나머지 원본 씬을
    다시 결합한 결과가 새 잡의 final 이 된다.
    현재 remix 를 지원하는 백엔드: sora-2 / sora-2-pro (OpenAI).
    """
    source = _get_job_or_404(request, job_id)
    settings = request.app.state.settings

    if source.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"완료된 잡만 remix 할 수 있습니다(status={source.status.value}).",
        )

    try:
        backend = get_backend(source.model, settings)
    except BackendNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not backend.supports_remix:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{source.model}' 백엔드는 remix 를 지원하지 않습니다. "
                "현재 지원: sora-2, sora-2-pro"
            ),
        )

    state = next(
        (s for s in source.scenes if s.index == body.scene_index), None
    )
    if state is None:
        raise HTTPException(
            status_code=422,
            detail=f"scene_index {body.scene_index} 가 범위를 벗어났습니다.",
        )
    if not state.backend_job_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "이 씬에는 백엔드 비디오 ID 가 없어 remix 할 수 없습니다 "
                "(remix 기능 추가 이전에 생성된 잡일 수 있습니다)."
            ),
        )

    manager = request.app.state.job_manager
    orchestrator = request.app.state.orchestrator
    job = manager.create(
        mode="remix", model=source.model,
        request={
            "source_job_id": source.id,
            "scene_index": body.scene_index,
            "prompt": body.prompt,
        },
    )
    manager.start(
        job,
        lambda: orchestrator.run_remix_job(
            job, source, body.scene_index, body.prompt
        ),
    )
    return JobCreatedResponse(job_id=job.id, status=job.status.value)


@router.get("/{job_id}/subtitles")
async def download_subtitles(request: Request, job_id: str):
    job = _get_job_or_404(request, job_id)
    if not job.subtitles_path or not Path(job.subtitles_path).exists():
        raise HTTPException(status_code=404, detail="자막 파일이 없습니다.")
    return FileResponse(
        Path(job.subtitles_path),
        media_type="application/x-subrip", filename=f"{job_id}.srt",
    )
