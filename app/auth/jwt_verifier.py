"""
auth/jwt_verifier.py
--------------------
auth-server(OAuth 2.1) 가 발급한 액세스 토큰(JWT, RS256)을 JWKS 공개키로
자체 검증한다. auth-server 의 examples/backend/main.go 검증 로직을 그대로 옮긴다.

검증 항목(참조 백엔드와 동일):
  - 서명: <ISSUER>/jwks.json 의 공개키 중 토큰 헤더 kid 에 맞는 키로 검증
  - 알고리즘: RS256 만 허용(alg=none / HS 혼동 공격 차단)
  - iss: 설정된 발급자와 정확히 일치
  - exp / nbf: 만료/활성시각(작은 leeway 허용)
  - (추가) aud: 설정 시에만 검증(미설정이면 검증 생략 — 참조 백엔드도 미검증)
  - (추가) scope: 필요한 스코프(기본 'api') 포함 여부

JWKS 는 PyJWKClient 가 가져와 캐시하고, kid 미스 시 자동 갱신한다
(Go 의 keyfunc 와 동일 동작). 라이브러리 시그니처는 Context7 로 확인(CLAUDE.md §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Principal:
    """검증된 호출자 정보(라우터에서 필요 시 사용)."""

    subject: str                       # sub (auth-server 의 user id) 또는 'app-key'
    username: Optional[str] = None     # username 클레임
    scopes: tuple[str, ...] = ()       # 공백 구분 scope → 튜플
    client_id: Optional[str] = None    # aud(클라이언트 id)
    auth_method: str = "jwt"           # "jwt" | "app_key"
    claims: dict = field(default_factory=dict)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


class AuthError(Exception):
    """토큰 검증 실패. status/code 로 HTTP 응답을 구성한다."""

    def __init__(self, message: str, *, status: int = 401,
                 code: str = "invalid_token") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


class JwtVerifier:
    """
    JWKS 기반 RS256 JWT 검증기.

    Parameters
    ----------
    issuer : str            iss 클레임으로 정확히 일치해야 하는 발급자 URL
    jwks_url : str          공개키(JWKS) 엔드포인트(컨테이너 내부 호스트 가능)
    audience : str          빈 문자열이면 aud 검증 생략
    required_scope : str    빈 문자열이면 scope 검증 생략(기본 'api')
    leeway_sec : int        exp/nbf 시계 오차 허용(초)
    cache_lifespan_sec : int  JWKS 캐시 TTL(초)
    """

    def __init__(
        self,
        issuer: str,
        jwks_url: str,
        *,
        audience: str = "",
        required_scope: str = "api",
        leeway_sec: int = 5,
        cache_lifespan_sec: int = 300,
    ) -> None:
        import jwt
        from jwt import PyJWKClient

        self._jwt = jwt
        self.issuer = issuer
        self.audience = audience or None
        self.required_scope = (required_scope or "").strip()
        self.leeway_sec = leeway_sec
        # PyJWKClient: kid 로 키 선택, 캐시 + 미스 시 자동 갱신(Context7 확인).
        self._client = PyJWKClient(
            jwks_url, cache_keys=True, lifespan=cache_lifespan_sec
        )

    # ------------------------------------------------------------------ #
    def verify(self, token: str) -> Principal:
        """토큰을 검증하고 Principal 을 반환한다. 실패 시 AuthError."""
        jwt = self._jwt

        # 1) 토큰 헤더 kid 에 맞는 공개키를 JWKS 에서 선택(필요 시 원격 갱신)
        try:
            signing_key = self._client.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError as exc:
            raise AuthError(f"서명 키를 찾을 수 없습니다: {exc}")
        except Exception as exc:  # noqa: BLE001 - 네트워크/파싱 등
            raise AuthError(f"JWKS 조회 실패: {exc}")

        # 2) 서명/표준 클레임 검증
        options = {"require": ["exp", "iss"]}
        kwargs = dict(
            algorithms=["RS256"],          # RS256 만 허용(중요: alg 혼동 차단)
            issuer=self.issuer,
            leeway=self.leeway_sec,
        )
        if self.audience:
            kwargs["audience"] = self.audience
        else:
            options["verify_aud"] = False

        try:
            claims = jwt.decode(token, signing_key.key, options=options, **kwargs)
        except jwt.ExpiredSignatureError:
            raise AuthError("토큰이 만료되었습니다")
        except jwt.ImmatureSignatureError:
            raise AuthError("토큰이 아직 활성화되지 않았습니다(nbf)")
        except jwt.InvalidIssuerError:
            raise AuthError("발급자(iss)가 일치하지 않습니다")
        except jwt.InvalidAudienceError:
            raise AuthError("대상(aud)이 일치하지 않습니다")
        except jwt.MissingRequiredClaimError as exc:
            raise AuthError(f"필수 클레임 누락: {exc}")
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"토큰 검증 실패: {exc}")

        # 3) 스코프 검증(공백 구분)
        scopes = tuple((claims.get("scope") or "").split())
        if self.required_scope and self.required_scope not in scopes:
            raise AuthError(
                f"필요한 스코프 '{self.required_scope}' 가 토큰에 없습니다",
                status=403, code="insufficient_scope",
            )

        # aud 는 문자열 또는 배열일 수 있다(auth-server 는 [client_id]).
        aud = claims.get("aud")
        if isinstance(aud, list):
            client_id = aud[0] if aud else None
        elif isinstance(aud, str):
            client_id = aud
        else:
            client_id = None

        return Principal(
            subject=claims.get("sub", ""),
            username=claims.get("username"),
            scopes=scopes,
            client_id=client_id,
            auth_method="jwt",
            claims=claims,
        )
