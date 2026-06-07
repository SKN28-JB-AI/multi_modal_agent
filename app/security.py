"""
security.py
-----------
앱 키(X-App-Key 헤더) 기반 인증.

[설계 노트]
- 프론트엔드는 모든 보호 엔드포인트 호출 시 X-App-Key 헤더를 보내야 한다.
- 키 비교는 secrets.compare_digest 로 타이밍 공격을 방지한다.
- 추후 JWT/사용자 인증으로 확장할 때 이 모듈만 교체하면 된다.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request


async def require_app_key(
    request: Request,
    x_app_key: str | None = Header(default=None, alias="X-App-Key"),
) -> str:
    """X-App-Key 헤더를 검증하는 FastAPI 의존성."""
    settings = request.app.state.settings

    if not x_app_key:
        raise HTTPException(
            status_code=401,
            detail="X-App-Key 헤더가 필요합니다.",
            headers={"WWW-Authenticate": "AppKey"},
        )

    for valid_key in settings.app_key_list:
        if secrets.compare_digest(x_app_key, valid_key):
            return x_app_key

    raise HTTPException(status_code=403, detail="유효하지 않은 앱 키입니다.")
