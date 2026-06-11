"""
작업 요청자(requested_by) 기록/노출 테스트.

- JWT 인증 사용자 → username/sub 기록, 응답에 requested_by 노출
- X-App-Key(서비스 호출) → 기록 없음(None), 응답에도 null
"""

from __future__ import annotations

from app.ads.schemas import AdJob, AdStoryboardOptions
from app.auth import Principal
from app.coupons import require_video_coupon
from app.security import require_auth, requester_of

from .conftest import auth_headers


def _jwt_principal() -> Principal:
    return Principal(
        subject="user-uuid-1", username="gunwoo",
        scopes=("api",), auth_method="jwt",
    )


# ---------------------------------------------------------------------- #
# 단위
# ---------------------------------------------------------------------- #
def test_requester_of_jwt_user():
    assert requester_of(_jwt_principal()) == ("gunwoo", "user-uuid-1")


def test_requester_of_app_key_is_empty():
    p = Principal(subject="app-key", username=None, auth_method="app_key")
    assert requester_of(p) == (None, None)


def test_requester_of_jwt_without_username():
    # username 클레임이 없는 토큰: 이름은 없고 id 만 기록
    p = Principal(subject="user-2", username=None, auth_method="jwt")
    assert requester_of(p) == (None, "user-2")


def test_ad_job_model_defaults():
    job = AdJob(id="x", prompt="p", options=AdStoryboardOptions())
    assert job.requested_by is None and job.requested_by_id is None


# ---------------------------------------------------------------------- #
# API: v1 메시지 모드
# ---------------------------------------------------------------------- #
def test_message_job_records_jwt_requester(client):
    app = client.app
    app.dependency_overrides[require_auth] = _jwt_principal
    # 쿠폰 차감은 auth-server 호출이 필요하므로 테스트에선 통과 처리
    app.dependency_overrides[require_video_coupon] = _jwt_principal
    try:
        res = client.post(
            "/v1/videos/message",
            json={"prompt": "test ad", "model": "mock"},
        )
        assert res.status_code == 202, res.text
        job_id = res.json()["job_id"]
        detail = client.get(f"/v1/jobs/{job_id}").json()
        assert detail["requested_by"] == "gunwoo"
    finally:
        app.dependency_overrides.clear()


def test_message_job_app_key_has_no_requester(client):
    res = client.post(
        "/v1/videos/message",
        json={"prompt": "test ad", "model": "mock"},
        headers=auth_headers(),
    )
    assert res.status_code == 202, res.text
    job_id = res.json()["job_id"]
    detail = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()
    assert detail["requested_by"] is None


def test_job_persists_requester(client, tmp_path):
    # manager 수준: 저장/복원 시 요청자 필드 유지
    manager = client.app.state.job_manager
    job = manager.create(
        mode="message", model="mock", request={},
        requested_by="gunwoo", requested_by_id="user-uuid-1",
    )
    raw = (manager.job_dir(job.id) / "job.json").read_text(encoding="utf-8")
    assert '"requested_by": "gunwoo"' in raw
    assert '"requested_by_id": "user-uuid-1"' in raw
