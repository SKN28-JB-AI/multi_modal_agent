"""
routers/ads.py
--------------
광고 제작 4단계 파이프라인 API (기존 /v1 비디오 API 와 독립).

  1) POST /v2/ads/storyboards          프롬프트 → 스토리보드 JSON (잡 생성)
  2) POST /v2/ads/{job_id}/images      컷별 첫 장면 이미지 (1 완료 후)
  3) POST /v2/ads/{job_id}/videos      이미지 기반 컷 비디오 + 결합 (2 완료 후 ★)
  4) POST /v2/ads/{job_id}/proposal    광고 기획서 PDF (2·3과 무관, 1 완료 후)

  GET  /v2/ads                         잡 목록
  GET  /v2/ads/image-models            이미지 모델 목록(벤더 3종)
  GET  /v2/ads/{job_id}                잡 상태 전체
  GET  /v2/ads/{job_id}/storyboard     스토리보드 JSON
  GET  /v2/ads/{job_id}/images/{cut}   컷 이미지 다운로드
  GET  /v2/ads/{job_id}/videos/{cut}   컷 클립 다운로드
  GET  /v2/ads/{job_id}/video          최종 결합본 다운로드
  GET  /v2/ads/{job_id}/proposal       기획서 PDF 다운로드

상태 코드 규약
  202 접수됨 / 404 잡·산출물 없음 / 409 이미 실행 중·완료(force 필요)
  412 선행 단계 미완료(★ images 미완료 상태에서 videos 요청)
  422 알 수 없는 모델 / 503 백엔드 API 키 미설정
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from ..ads.image_backends import (
    ImageBackendNotConfigured,
    available_image_models,
    get_image_backend,
    image_model_info,
)
from ..ads.images import ImagesStageError, run_images_stage
from ..ads.manager import AdJobManager
from ..ads.pdf_gen import generate_proposal_pdf
from ..ads.schemas import (
    AdImagesRequest,
    AdJob,
    AdJobAccepted,
    AdJobStatusResponse,
    AdStoryboard,
    AdVideosRequest,
    AdStoryboardRequest,
    CutAssetOut,
    ImageModelsResponse,
    STAGE_NAMES,
    StageStateOut,
)
from ..ads.storyboard import generate_storyboard
from ..ads.videos import VideosStageError, run_videos_stage
from ..backends import BackendNotConfigured, available_models, get_backend
from ..security import require_app_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v2/ads", tags=["ads"], dependencies=[Depends(require_app_key)]
)


# ====================================================================== #
# 내부 헬퍼
# ====================================================================== #
def _manager(request: Request) -> AdJobManager:
    return request.app.state.ad_job_manager


def _accepted(job_id: str, stage: str) -> JSONResponse:
    body = AdJobAccepted(
        job_id=job_id,
        stage=stage,
        status="in_progress",
        status_url=f"/v2/ads/{job_id}",
        message=(
            f"'{stage}' 단계가 접수되었습니다. "
            f"status_url 로 진행 상태를 조회하세요."
        ),
    )
    return JSONResponse(status_code=202, content=body.model_dump())


def _resolve_image_backend(request: Request, model: str):
    """이미지 모델 검증 + 백엔드 생성. 실패 시 422/503."""
    settings = request.app.state.settings
    try:
        return get_image_backend(model, settings)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"알 수 없는 이미지 모델 '{model}' 입니다. "
                f"사용 가능: {', '.join(sorted(available_image_models()))}"
            ),
        )
    except ImageBackendNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _resolve_video_backend(request: Request, model: str):
    """비디오 모델 검증 + image-to-video 지원 확인. 실패 시 422/503."""
    settings = request.app.state.settings
    try:
        backend = get_backend(model, settings)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"알 수 없는 비디오 모델 '{model}' 입니다. "
                f"사용 가능: {', '.join(sorted(available_models()))}"
            ),
        )
    except BackendNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not backend.supports_image_input:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{model}' 모델은 이미지 기반 생성(image-to-video)을 "
                f"지원하지 않습니다."
            ),
        )
    return backend


def _completed_image_paths(job: AdJob) -> dict[int, Path]:
    return {
        a.cut: Path(a.path)
        for a in job.images
        if a.status == "completed" and a.path and Path(a.path).exists()
    }


def _file_or_404(path_str: str | None, what: str) -> Path:
    if not path_str:
        raise HTTPException(status_code=404, detail=f"{what}이(가) 아직 없습니다.")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"{what} 파일을 찾을 수 없습니다."
        )
    return path


# ====================================================================== #
# 메타 조회
# ====================================================================== #
@router.get("/image-models", response_model=ImageModelsResponse)
async def list_image_models(request: Request):
    """사용 가능한 이미지 모델(벤더 3종) 목록."""
    settings = request.app.state.settings
    return ImageModelsResponse(
        models=image_model_info(settings, settings.ad_image_model_default)
    )


# ====================================================================== #
# 1단계: 스토리보드 생성
# ====================================================================== #
@router.post("/storyboards", response_model=AdJobAccepted, status_code=202)
async def create_storyboard(request: Request, body: AdStoryboardRequest):
    """프롬프트로 스토리보드 JSON 을 생성한다(잡 생성, 202 + 폴링)."""
    settings = request.app.state.settings
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY 가 설정되지 않아 스토리보드를 생성할 수 없습니다.",
        )

    manager = _manager(request)
    job = manager.create(prompt=body.prompt, options=body.options)
    manager.begin_stage(job, "storyboard")

    async def _run() -> None:
        try:
            sb = await generate_storyboard(settings, job.prompt, job.options)
            manager.update(job, storyboard=sb)
            manager.finish_stage(job, "storyboard")
        except Exception as exc:  # noqa: BLE001
            logger.exception("[ads] storyboard 단계 실패: job=%s", job.id)
            manager.finish_stage(job, "storyboard", error=str(exc))

    manager.start(job, "storyboard", _run)
    return _accepted(job.id, "storyboard")


# ====================================================================== #
# 2단계: 컷별 첫 장면 이미지 생성
# ====================================================================== #
@router.post("/{job_id}/images", response_model=AdJobAccepted, status_code=202)
async def create_cut_images(
    request: Request,
    job_id: str,
    body: AdImagesRequest | None = None,
    force: bool = Query(default=False, description="완료된 단계 재실행 허용"),
):
    """스토리보드의 컷마다 첫 장면 이미지를 생성한다(1단계 완료 후)."""
    settings = request.app.state.settings
    manager = _manager(request)
    job = manager.get_or_404(job_id)

    model = (body.model if body and body.model else None) \
        or settings.ad_image_model_default
    backend = _resolve_image_backend(request, model)   # 검증 먼저(실패 시 선점 안 함)
    manager.begin_stage(job, "images", force=force)    # 412/409 게이팅
    manager.update(job, image_model=model)

    async def _run() -> None:
        try:
            if job.storyboard is None:   # begin_stage 가 보장하지만 방어적으로
                raise RuntimeError("스토리보드가 없습니다(내부 상태 불일치).")
            assets = await run_images_stage(
                settings=settings,
                backend=backend,
                storyboard=job.storyboard,
                options=job.options,
                out_dir=manager.images_dir(job.id),
            )
            manager.update(job, images=assets)
            manager.finish_stage(job, "images")
        except ImagesStageError as exc:
            logger.exception("[ads] images 단계 실패: job=%s", job.id)
            manager.update(job, images=exc.partial_assets)  # 부분 산출물 보존
            manager.finish_stage(job, "images", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[ads] images 단계 실패: job=%s", job.id)
            manager.finish_stage(job, "images", error=str(exc))

    manager.start(job, "images", _run)
    return _accepted(job.id, "images")


# ====================================================================== #
# 3단계: 이미지 기반 컷 비디오 생성 (★ 2단계 완료 후에만 — 412)
# ====================================================================== #
@router.post("/{job_id}/videos", response_model=AdJobAccepted, status_code=202)
async def create_cut_videos(
    request: Request,
    job_id: str,
    body: AdVideosRequest | None = None,
    force: bool = Query(default=False, description="완료된 단계 재실행 허용"),
):
    """
    컷별 첫 장면 이미지를 시작 프레임으로 비디오를 생성하고 결합한다.
    images 단계가 완료되지 않았으면 412 를 반환한다.
    """
    settings = request.app.state.settings
    manager = _manager(request)
    job = manager.get_or_404(job_id)

    model = (body.model if body and body.model else None) \
        or settings.ad_video_model_default
    backend = _resolve_video_backend(request, model)   # 검증 먼저
    manager.begin_stage(job, "videos", force=force)    # ★ images 완료 강제(412)

    # begin_stage 통과 후에도 파일이 실제로 존재하는지 확인한다.
    image_paths = _completed_image_paths(job)
    missing = [
        c.cut for c in (job.storyboard.cuts if job.storyboard else [])
        if c.cut not in image_paths
    ]
    if missing:
        manager.finish_stage(
            job, "videos",
            error=f"컷 {', '.join(map(str, missing))} 이미지 파일이 없습니다.",
        )
        raise HTTPException(
            status_code=412,
            detail=(
                f"컷 {', '.join(map(str, missing))} 의 이미지 파일이 없습니다. "
                f"images 단계를 다시 실행하세요(?force=true)."
            ),
        )
    manager.update(job, video_model=model)

    async def _run() -> None:
        try:
            assert job.storyboard is not None
            assets, final_path = await run_videos_stage(
                settings=settings,
                backend=backend,
                storyboard=job.storyboard,
                options=job.options,
                image_paths=image_paths,
                out_dir=manager.videos_dir(job.id),
            )
            manager.update(job, videos=assets, final_video_path=final_path)
            manager.finish_stage(job, "videos")
        except VideosStageError as exc:
            logger.exception("[ads] videos 단계 실패: job=%s", job.id)
            manager.update(job, videos=exc.partial_assets)  # 부분 산출물 보존
            manager.finish_stage(job, "videos", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[ads] videos 단계 실패: job=%s", job.id)
            manager.finish_stage(job, "videos", error=str(exc))

    manager.start(job, "videos", _run)
    return _accepted(job.id, "videos")


# ====================================================================== #
# 4단계: 광고 기획서 PDF (2·3단계와 무관하게 실행 가능)
# ====================================================================== #
@router.post("/{job_id}/proposal", response_model=AdJobAccepted, status_code=202)
async def create_proposal(
    request: Request,
    job_id: str,
    force: bool = Query(default=False, description="완료된 단계 재실행 허용"),
):
    """
    스토리보드 기반 광고 기획서 PDF 를 생성한다.
    스토리보드(1단계)만 완료되어 있으면 되고, images/videos 와 무관하다.
    컷 이미지가 이미 생성돼 있으면 기획서에 함께 삽입된다.
    """
    settings = request.app.state.settings
    manager = _manager(request)
    job = manager.get_or_404(job_id)
    manager.begin_stage(job, "pdf", force=force)   # storyboard 완료만 요구

    async def _run() -> None:
        try:
            assert job.storyboard is not None
            out_path = manager.job_dir(job.id) / "proposal.pdf"
            await asyncio.to_thread(
                generate_proposal_pdf,
                settings,
                job.storyboard,
                out_path,
                _completed_image_paths(job),   # 있으면 삽입, 없어도 무방
                job.prompt,
            )
            manager.update(job, pdf_path=str(out_path))
            manager.finish_stage(job, "pdf")
        except Exception as exc:  # noqa: BLE001
            logger.exception("[ads] pdf 단계 실패: job=%s", job.id)
            manager.finish_stage(job, "pdf", error=str(exc))

    manager.start(job, "pdf", _run)
    return _accepted(job.id, "pdf")


# ====================================================================== #
# 상태/산출물 조회
# ====================================================================== #
def _status_response(job: AdJob) -> AdJobStatusResponse:
    base = f"/v2/ads/{job.id}"

    def _asset_out(asset, kind: str) -> CutAssetOut:
        url = None
        if asset.status == "completed" and asset.path:
            url = f"{base}/{kind}/{asset.cut}"
        return CutAssetOut(
            cut=asset.cut, status=asset.status, error=asset.error, url=url
        )

    return AdJobStatusResponse(
        job_id=job.id,
        prompt=job.prompt,
        options=job.options,
        image_model=job.image_model,
        video_model=job.video_model,
        stages={
            name: StageStateOut(**job.stage(name).model_dump())
            for name in STAGE_NAMES
        },
        storyboard=job.storyboard,
        images=[_asset_out(a, "images") for a in job.images],
        videos=[_asset_out(a, "videos") for a in job.videos],
        storyboard_url=(
            f"{base}/storyboard"
            if job.storyboard_stage.status == "completed" else None
        ),
        final_video_url=(
            f"{base}/video" if job.final_video_path else None
        ),
        pdf_url=f"{base}/proposal" if job.pdf_path else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("", response_model=list[AdJobStatusResponse])
async def list_ad_jobs(request: Request):
    """광고 파이프라인 잡 목록(최신순)."""
    return [_status_response(j) for j in _manager(request).list()]


@router.get("/{job_id}", response_model=AdJobStatusResponse)
async def get_ad_job(request: Request, job_id: str):
    """잡 상태 전체(단계별 상태 + 산출물 URL)."""
    return _status_response(_manager(request).get_or_404(job_id))


@router.get("/{job_id}/storyboard", response_model=AdStoryboard)
async def get_storyboard(request: Request, job_id: str):
    """1단계 산출물: 스토리보드 JSON."""
    job = _manager(request).get_or_404(job_id)
    if job.storyboard is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "스토리보드가 아직 없습니다. "
                f"(storyboard 단계 상태: {job.storyboard_stage.status})"
            ),
        )
    return job.storyboard


@router.get("/{job_id}/images/{cut}")
async def download_cut_image(request: Request, job_id: str, cut: int):
    """2단계 산출물: 컷 첫 장면 이미지(PNG)."""
    job = _manager(request).get_or_404(job_id)
    asset = next((a for a in job.images if a.cut == cut), None)
    if asset is None or asset.status != "completed":
        raise HTTPException(
            status_code=404, detail=f"컷 {cut} 이미지가 아직 없습니다."
        )
    path = _file_or_404(asset.path, f"컷 {cut} 이미지")
    return FileResponse(
        path, media_type="image/png",
        filename=f"{job_id}_cut_{cut:02d}.png",
    )


@router.get("/{job_id}/videos/{cut}")
async def download_cut_video(request: Request, job_id: str, cut: int):
    """3단계 산출물: 컷 클립(MP4)."""
    job = _manager(request).get_or_404(job_id)
    asset = next((a for a in job.videos if a.cut == cut), None)
    if asset is None or asset.status != "completed":
        raise HTTPException(
            status_code=404, detail=f"컷 {cut} 클립이 아직 없습니다."
        )
    path = _file_or_404(asset.path, f"컷 {cut} 클립")
    return FileResponse(
        path, media_type="video/mp4",
        filename=f"{job_id}_cut_{cut:02d}.mp4",
    )


@router.get("/{job_id}/video")
async def download_final_video(request: Request, job_id: str):
    """3단계 산출물: 최종 결합본(MP4)."""
    job = _manager(request).get_or_404(job_id)
    path = _file_or_404(job.final_video_path, "최종 결합본")
    return FileResponse(
        path, media_type="video/mp4", filename=f"{job_id}_final.mp4"
    )


@router.get("/{job_id}/proposal")
async def download_proposal(request: Request, job_id: str):
    """4단계 산출물: 광고 기획서 PDF."""
    job = _manager(request).get_or_404(job_id)
    path = _file_or_404(job.pdf_path, "기획서 PDF")
    return FileResponse(
        path, media_type="application/pdf",
        filename=f"{job_id}_proposal.pdf",
    )
