"""GET /v1/models — 백엔드 레지스트리 노출/확장성 테스트."""

from .conftest import auth_headers


def test_models_lists_registered_backends(client):
    resp = client.get("/v1/models", headers=auth_headers())
    names = {m["name"] for m in resp.json()["models"]}
    # 기본 등록 백엔드들
    assert {"sora-2", "sora-2-pro", "veo-3.1", "veo-3.1-fast",
            "ltx-2.3", "ltx-2.3-fast"} <= names
    # 테스트에서 동적으로 추가한 백엔드도 노출 — 확장성 검증
    assert "mock" in names


def test_configured_flag_reflects_api_keys(make_client):
    client = make_client(openai_api_key="", gemini_api_key="", fal_api_key="")
    models = {
        m["name"]: m
        for m in client.get("/v1/models", headers=auth_headers()).json()["models"]
    }
    assert models["sora-2"]["configured"] is False
    assert models["veo-3.1"]["configured"] is False
    assert models["ltx-2.3"]["configured"] is False
    assert models["mock"]["configured"] is True


def test_backend_durations_exposed(client):
    models = {
        m["name"]: m
        for m in client.get("/v1/models", headers=auth_headers()).json()["models"]
    }
    assert models["sora-2"]["supported_durations"] == [4.0, 8.0, 12.0]
    assert 20.0 in models["ltx-2.3-fast"]["supported_durations"]
