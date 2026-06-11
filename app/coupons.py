"""
coupons.py
----------
사용 횟수 제한(쿠폰) 차감 의존성.

auth-server 가 사용자별 잔여 쿠폰(video/ad)을 보관하며, 작업 생성 엔드포인트는
이 의존성으로 **사용자 본인의 Bearer 토큰을 그대로 전달**하여 1개를 차감한다.

정책
- JWT 인증 사용자만 차감 대상. X-App-Key(서비스 간/내부) 호출은 쿠폰 미적용.
- 관리자(admin 클레임)는 auth-server 가 차감 없이 unlimited 로 응답.
- 잔여 0 → 402 Payment Required (사용자에게 발급 요청 안내).
- auth-server 연결 불가 → 503 (fail-closed: 제한 우회 방지).
- 차감 시점은 작업 "생성" — 작업 실패 시 환불하지 않는다.
"""

from __future__ import annotations

from typing import Optional

import httpx
from fastapi import Depends, Header, HTTPException, Request

from .auth import Principal
from .security import require_auth

_LABELS = {"video": "영상 생성", "ad": "광고 파이프라인"}


def _coupon_base(settings) -> str:
    """auth-server 베이스 URL — JWKS_URL 에서 유도(같은 도커 네트워크 호스트)."""
    jwks = getattr(settings, "jwks_url", "") or ""
    if jwks.endswith("/jwks.json"):
        return jwks[: -len("/jwks.json")]
    return (getattr(settings, "auth_issuer", "") or "").rstrip("/")


def require_coupon(coupon_type: str):
    """coupon_type("video"|"ad") 쿠폰 1개를 차감하는 FastAPI 의존성 팩토리."""

    async def dependency(
        request: Request,
        principal: Principal = Depends(require_auth),  # 요청당 1회 캐시됨
        authorization: Optional[str] = Header(default=None),
    ) -> Principal:
        # 앱 키 인증(서비스 간 호출)은 사용자 단위 제한 대상이 아니다.
        if principal.auth_method != "jwt":
            return principal

        base = _coupon_base(request.app.state.settings)
        if not base:
            # JWT 가 검증됐다면 auth-server 설정이 있는 상태 — 방어적 분기
            raise HTTPException(503, "쿠폰 서버(auth-server) 설정이 없습니다.")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(
                    base + "/coupons/consume",
                    json={"type": coupon_type},
                    headers={"Authorization": authorization},
                )
        except httpx.HTTPError:
            raise HTTPException(503, "쿠폰 서버(auth-server)에 연결할 수 없습니다.")

        if res.status_code == 402:
            label = _LABELS.get(coupon_type, coupon_type)
            raise HTTPException(
                402,
                f"'{label}' 잔여 쿠폰이 없습니다. 관리자에게 쿠폰 발급을 요청하세요.",
            )
        if res.status_code == 401:
            raise HTTPException(
                401, "쿠폰 차감 중 토큰 검증에 실패했습니다.",
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )
        if res.status_code >= 400:
            raise HTTPException(503, f"쿠폰 차감 실패 (auth-server {res.status_code})")
        return principal

    return dependency


require_video_coupon = require_coupon("video")
require_ad_coupon = require_coupon("ad")
