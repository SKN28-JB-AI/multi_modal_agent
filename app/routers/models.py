"""
routers/models.py
-----------------
GET /v1/models : 등록된 비디오 백엔드 목록 + 사용 가능 여부.
프론트엔드가 모델 선택 UI 를 구성할 때 사용한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..backends import backend_info
from ..schemas import BackendInfo, ModelsResponse
from ..security import require_auth

router = APIRouter(
    prefix="/v1", tags=["models"], dependencies=[Depends(require_auth)]
)


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request):
    infos = backend_info(request.app.state.settings)
    return ModelsResponse(models=[BackendInfo(**i) for i in infos])
