"""
routers/logos.py
----------------
서버 logos/ 폴더의 브랜드 로고 관리.

  GET /v1/logos : 사용 가능한 로고 파일 목록 + 기본 선택 로고

광고 영상 생성(PDF 모드) 시 로고 적용 우선순위:
  1) 요청 multipart 의 logo 업로드 (일회성 로고)
  2) options.logo_name 으로 지정한 logos/ 폴더의 파일
  3) logos/default.png 가 있으면 그것
  4) logos/ 의 첫 파일(이름순)
  (폴더가 비어 있고 업로드도 없으면 로고 없이 진행)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..security import require_app_key

router = APIRouter(
    prefix="/v1", tags=["logos"], dependencies=[Depends(require_app_key)]
)

LOGO_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def list_logo_files(logos_dir: Path) -> list[Path]:
    if not logos_dir.is_dir():
        return []
    return sorted(
        p for p in logos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in LOGO_EXTS
    )


def resolve_logo(logos_dir: Path, logo_name: Optional[str]) -> Optional[Path]:
    """
    logos/ 폴더에서 적용할 로고를 결정한다.
    - logo_name 지정: 해당 파일(없거나 경로 조작 시 HTTPException 422)
    - 미지정: default.png > 첫 파일 > None
    """
    if logo_name:
        # 경로 조작 방지: 파일명만 허용
        if Path(logo_name).name != logo_name:
            raise HTTPException(
                status_code=422, detail="logo_name 은 파일명만 허용됩니다."
            )
        candidate = logos_dir / logo_name
        if not candidate.is_file():
            available = ", ".join(p.name for p in list_logo_files(logos_dir))
            raise HTTPException(
                status_code=422,
                detail=(
                    f"로고 '{logo_name}' 가 없습니다. "
                    f"사용 가능: {available or '(없음)'}"
                ),
            )
        return candidate

    files = list_logo_files(logos_dir)
    if not files:
        return None
    default = logos_dir / "default.png"
    return default if default.is_file() else files[0]


class LogosResponse(BaseModel):
    logos: list[str]
    default: Optional[str] = None   # 미지정 시 자동 적용될 로고


@router.get("/logos", response_model=LogosResponse)
async def list_logos(request: Request):
    logos_dir = Path(request.app.state.settings.logos_dir)
    files = list_logo_files(logos_dir)
    default = resolve_logo(logos_dir, None)
    return LogosResponse(
        logos=[p.name for p in files],
        default=default.name if default else None,
    )
