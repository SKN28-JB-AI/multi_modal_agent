"""
security.py
-----------
인증: auth-server(OAuth 2.1) 발급 **Bearer JWT** 를 우선 검증하고,
없으면 기존 **X-App-Key**(정적 앱 키)로 폴백한다(둘 다 허용).

[설계 노트]
- JWT: <ISSUER>/jwks.json 공개키로 자체검증(RS256, iss, exp/nbf, scope).
  검증 로직은 app/auth/jwt_verifier.py (auth-server examples/backend 미러).
- X-App-Key: 서비스간 호출/기존 프론트 호환용. secrets.compare_digest 로
  타이밍 공격 방지. 검증된 앱 키에는 'api' 스코프를 부여한 것으로 간주한다.
- 보호 라우터는 Depends(require_auth) 를 사용한다. require_app_key 는
  하위호환을 위해 유지(앱 키 전용 검사).
"""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from .auth import AuthError, Principal


def requester_of(principal: Principal) -> tuple[Optional[str], Optional[str]]:
    """
    Principal 에서 잡 기록용 (표시이름, 사용자ID) 를 뽑는다.

    JWT 인증 사용자만 기록하며, 앱 키 호출(서비스간/스크립트)은
    사용자 정보가 없으므로 (None, None) — 프론트에는 표시되지 않는다.
    """
    if principal.auth_method != "jwt":
        return None, None
    return principal.username or None, principal.subject or None


def _app_key_principal() -> Principal:
    """검증된 정적 앱 키 호출자(스코프 'api' 부여)."""
    return Principal(
        subject="app-key", username=None, scopes=("api",),
        client_id=None, auth_method="app_key",
    )


async def require_auth(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_app_key: Optional[str] = Header(default=None, alias="X-App-Key"),
) -> Principal:
    """
    Bearer JWT(우선) 또는 X-App-Key(폴백) 를 검증하는 FastAPI 의존성.
    실패 시 401(인증)/403(권한)으로 응답한다.
    """
    settings = request.app.state.settings
    verifier = getattr(request.app.state, "jwt_verifier", None)

    # 1) Authorization: Bearer <jwt> 우선
    if authorization and authorization.strip().lower().startswith("bearer "):
        token = authorization.split(None, 1)[1].strip()
        if verifier is None:
            raise HTTPException(
                status_code=401,
                detail=(
                    "JWT 인증이 구성되지 않았습니다(AUTH_ISSUER/JWKS_URL 미설정). "
                    "X-App-Key 를 사용하세요."
                ),
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )
        try:
            return verifier.verify(token)
        except AuthError as exc:
            raise HTTPException(
                status_code=exc.status,
                detail=exc.message,
                headers={"WWW-Authenticate": f'Bearer error="{exc.code}"'},
            )

    # 2) X-App-Key 폴백
    if x_app_key:
        for valid_key in settings.app_key_list:
            if secrets.compare_digest(x_app_key, valid_key):
                return _app_key_principal()
        raise HTTPException(status_code=403, detail="유효하지 않은 앱 키입니다.")

    # 3) 둘 다 없음
    raise HTTPException(
        status_code=401,
        detail="인증이 필요합니다: Authorization Bearer 토큰 또는 X-App-Key 헤더.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_app_key(
    request: Request,
    x_app_key: Optional[str] = Header(default=None, alias="X-App-Key"),
) -> str:
    """[하위호환] X-App-Key 전용 검증 의존성."""
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
