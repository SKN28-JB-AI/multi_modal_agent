"""
routers/videos.py
-----------------
동영상 생성 요청 엔드포인트 (모드별 분리).

  POST /v1/videos/message : 메시지(단일 프롬프트) 모드
  POST /v1/videos/pdf     : PDF 기획서 모드 (multipart)

둘 다 202 + job_id 를 즉시 반환하고, 생성은 백그라운드에서 진행된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Request, UploadFile,
)
from pydantic import ValidationError

from ..backends import BackendNotConfigured, available_models, get_backend
from ..pipeline.orchestrator import Orchestrator
from ..schemas import JobCreatedResponse, MessageRequest, PdfJobOptions
from ..security import require_app_key

router = APIRouter(prefix="/v1/videos", tags=["videos"])

MAX_PDF_BYTES = 30 * 1024 * 1024   # 30MB
MAX_LOGO_BYTES = 5 * 1024 * 1024   # 5MB


def _validate_model(request: Request, model: str) -> None:
    """모델 이름 검증 + 키 설정 여부 확인. 실패 시 4xx/503."""
    settings = request.app.state.settings
    try:
        get_backend(model, settings)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"알 수 없는 모델 '{model}' 입니다. "
                f"사용 가능: {', '.join(sorted(available_models()))}"
            ),
        )
    except BackendNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------- #
# 메시지 모드
# ---------------------------------------------------------------------- #
@router.post(
    "/message",
    response_model=JobCreatedResponse,
    status_code=202,
    dependencies=[Depends(require_app_key)],
)
async def create_message_job(request: Request, body: MessageRequest):
    _validate_model(request, body.model)

    manager = request.app.state.job_manager
    orchestrator: Orchestrator = request.app.state.orchestrator

    job = manager.create(
        mode="message", model=body.model, request=body.model_dump()
    )
    manager.start(job, lambda: orchestrator.run_message_job(job))
    return JobCreatedResponse(job_id=job.id, status=job.status.value)


# ---------------------------------------------------------------------- #
# PDF 기획서 모드
# ---------------------------------------------------------------------- #
@router.post(
    "/pdf",
    response_model=JobCreatedResponse,
    status_code=202,
    dependencies=[Depends(require_app_key)],
)
async def create_pdf_job(
    request: Request,
    file: UploadFile = File(..., description="광고 기획서 PDF"),
    model: str = Form(..., description="비디오 백엔드 이름"),
    options: Optional[str] = Form(
        default=None, description="PdfJobOptions JSON 문자열"
    ),
    logo: Optional[UploadFile] = File(
        default=None, description="오버레이할 로고 PNG(선택)"
    ),
):
    _validate_model(request, model)

    # PDF 모드는 파싱/스토리보드에 OpenAI LLM 이 필요하다.
    settings = request.app.state.settings
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="PDF 모드에는 OPENAI_API_KEY 설정이 필요합니다(LLM 파싱).",
        )

    # 옵션 파싱
    try:
        opts = (
            PdfJobOptions.model_validate_json(options)
            if options else PdfJobOptions()
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"options 형식 오류: {exc}")

    # 파일 검증
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=422, detail="PDF 파일만 업로드할 수 있습니다.")
    pdf_bytes = await file.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="유효한 PDF 가 아닙니다.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF 가 30MB 를 초과합니다.")

    manager = request.app.state.job_manager
    orchestrator: Orchestrator = request.app.state.orchestrator

    job = manager.create(
        mode="pdf", model=model,
        request={"filename": file.filename, "options": opts.model_dump()},
    )
    job_dir = manager.job_dir(job.id)
    pdf_path = job_dir / "input.pdf"
    pdf_path.write_bytes(pdf_bytes)

    logo_path: Optional[Path] = None
    if logo is not None:
        logo_bytes = await logo.read()
        if len(logo_bytes) > MAX_LOGO_BYTES:
            raise HTTPException(status_code=413, detail="로고가 5MB 를 초과합니다.")
        if logo_bytes:
            logo_path = job_dir / "logo.png"
            logo_path.write_bytes(logo_bytes)

    manager.start(
        job, lambda: orchestrator.run_pdf_job(job, pdf_path, opts, logo_path)
    )
    return JobCreatedResponse(job_id=job.id, status=job.status.value)
