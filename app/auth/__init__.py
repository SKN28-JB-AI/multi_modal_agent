"""
auth 패키지: auth-server(OAuth 2.1) 발급 JWT 의 자체검증 로직.

auth-server 는 RS256 으로 액세스 토큰을 서명하고 공개키를 /jwks.json 으로
노출한다. 본 서비스는 인증서버를 매 요청마다 호출하지 않고 JWKS 공개키로
토큰을 '자체 검증'한다(examples/backend 의 Go 구현과 동일 원리).
"""

from .jwt_verifier import AuthError, JwtVerifier, Principal

__all__ = ["AuthError", "JwtVerifier", "Principal"]
