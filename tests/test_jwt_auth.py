"""
auth-server(OAuth 2.1) 발급 JWT 의 자체검증 테스트.

네트워크 없이 검증하기 위해, RSA 키쌍을 만들고 PyJWKClient 를 가짜 클라이언트로
교체한다(공개키를 바로 반환). auth-server examples/backend 의 검증 규칙
(RS256 / iss 일치 / exp / scope)을 그대로 따른다.
"""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.auth import AuthError, JwtVerifier
from .conftest import APP_KEY, auth_headers


ISSUER = "https://auth.test"
KID = "test-kid"


# ---------------------------------------------------------------------- #
# 키/토큰/가짜 JWKS 유틸
# ---------------------------------------------------------------------- #
_PRIV = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIV_PEM = _PRIV.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
_PUB = _PRIV.public_key()


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJWKClient:
    """kid 와 무관하게 테스트 공개키를 반환(서명 검증은 실제로 수행됨)."""

    def __init__(self, public_key):
        self._pub = public_key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._pub)


def _mint(**overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": "user-1",
        "username": "demo",
        "scope": "openid profile api",
        "aud": ["frontend-spa"],
        "iat": now,
        "nbf": now,
        "exp": now + 300,
    }
    claims.update(overrides.pop("claims", {}))
    alg = overrides.pop("alg", "RS256")
    key = overrides.pop("key", _PRIV_PEM)
    return pyjwt.encode(claims, key, algorithm=alg, headers={"kid": KID})


def _verifier(**kw) -> JwtVerifier:
    v = JwtVerifier(issuer=ISSUER, jwks_url="https://auth.test/jwks.json", **kw)
    v._client = _FakeJWKClient(_PUB)   # 네트워크 차단
    return v


# ---------------------------------------------------------------------- #
# 단위 테스트: 검증기
# ---------------------------------------------------------------------- #
def test_valid_token_passes():
    p = _verifier().verify(_mint())
    assert p.subject == "user-1"
    assert p.username == "demo"
    assert "api" in p.scopes
    assert p.client_id == "frontend-spa"
    assert p.auth_method == "jwt"


def test_expired_token_rejected():
    with pytest.raises(AuthError) as e:
        _verifier().verify(_mint(claims={"exp": int(time.time()) - 10}))
    assert e.value.status == 401


def test_wrong_issuer_rejected():
    with pytest.raises(AuthError):
        _verifier().verify(_mint(claims={"iss": "https://evil.test"}))


def test_missing_scope_rejected_403():
    with pytest.raises(AuthError) as e:
        _verifier().verify(_mint(claims={"scope": "openid profile"}))
    assert e.value.status == 403
    assert e.value.code == "insufficient_scope"


def test_scope_check_can_be_disabled():
    # required_scope="" 이면 scope 없이도 통과
    p = _verifier(required_scope="").verify(_mint(claims={"scope": "openid"}))
    assert p.subject == "user-1"


def test_non_rs256_rejected():
    # HS256 으로 서명한 토큰은 algorithms=["RS256"] 에서 거부되어야 한다.
    bad = pyjwt.encode({"iss": ISSUER, "sub": "x", "scope": "api",
                        "exp": int(time.time()) + 300},
                       "shared-secret", algorithm="HS256",
                       headers={"kid": KID})
    with pytest.raises(AuthError):
        _verifier().verify(bad)


def test_audience_validated_when_configured():
    v = _verifier(audience="frontend-spa")
    assert v.verify(_mint()).client_id == "frontend-spa"   # 일치 → 통과
    with pytest.raises(AuthError):
        v.verify(_mint(claims={"aud": ["other-client"]}))   # 불일치 → 거부


def test_missing_exp_rejected():
    # exp 누락은 require 규칙으로 거부
    tok = pyjwt.encode({"iss": ISSUER, "sub": "x", "scope": "api"},
                       _PRIV_PEM, algorithm="RS256", headers={"kid": KID})
    with pytest.raises(AuthError):
        _verifier().verify(tok)


# ---------------------------------------------------------------------- #
# E2E: FastAPI 의존성(JWT + X-App-Key 병행)
# ---------------------------------------------------------------------- #
def _with_jwt(client):
    """테스트용 검증기(가짜 JWKS)를 앱 상태에 주입한다."""
    client.app.state.jwt_verifier = _verifier()
    return client


def test_endpoint_accepts_bearer_jwt(make_client):
    client = _with_jwt(make_client())
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {_mint()}"})
    assert r.status_code == 200


def test_endpoint_rejects_expired_bearer(make_client):
    client = _with_jwt(make_client())
    tok = _mint(claims={"exp": int(time.time()) - 5})
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate", "").startswith("Bearer")


def test_endpoint_rejects_insufficient_scope(make_client):
    client = _with_jwt(make_client())
    tok = _mint(claims={"scope": "openid profile"})
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


def test_app_key_still_works_alongside_jwt(make_client):
    client = _with_jwt(make_client())
    assert client.get("/v1/models", headers=auth_headers()).status_code == 200


def test_no_auth_returns_401(make_client):
    client = _with_jwt(make_client())
    assert client.get("/v1/models").status_code == 401


def test_bearer_when_jwt_not_configured_returns_401(make_client):
    # auth_issuer 미설정 → verifier 없음 → Bearer 제시 시 401(미구성 안내)
    client = make_client()
    assert client.app.state.jwt_verifier is None
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {_mint()}"})
    assert r.status_code == 401


def test_settings_jwt_enabled_toggle():
    from app.config import Settings
    off = Settings(app_keys="k", _env_file=None)
    assert off.jwt_enabled is False
    on = Settings(app_keys="k", auth_issuer="http://localhost:9000", _env_file=None)
    assert on.jwt_enabled is True
    assert on.effective_jwks_url == "http://localhost:9000/jwks.json"
    # JWKS_URL 명시값이 우선(컨테이너 내부 호스트)
    on2 = Settings(app_keys="k", auth_issuer="http://localhost:9000",
                   jwks_url="http://auth-server:9000/jwks.json", _env_file=None)
    assert on2.effective_jwks_url == "http://auth-server:9000/jwks.json"
