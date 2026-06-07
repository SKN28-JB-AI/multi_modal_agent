"""앱 키 인증 테스트."""

from .conftest import auth_headers


def test_missing_app_key_returns_401(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_wrong_app_key_returns_403(client):
    resp = client.get("/v1/models", headers={"X-App-Key": "wrong-key"})
    assert resp.status_code == 403


def test_valid_app_key_returns_200(client):
    resp = client.get("/v1/models", headers=auth_headers())
    assert resp.status_code == 200


def test_health_is_public(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_app_refuses_to_start_without_app_keys(tmp_path):
    import pytest
    from app.config import Settings
    from app.main import create_app

    with pytest.raises(RuntimeError):
        create_app(Settings(app_keys="", data_dir=str(tmp_path), _env_file=None))
