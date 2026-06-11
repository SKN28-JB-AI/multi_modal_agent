"""
모델별 클립 길이 제한 + 기본값(8초) 테스트.

- 단위: validate_duration / normalize_duration / default_duration
- API : /v1/videos/message 의 범위 초과 422, 미지정 시 기본 8초 보정
- /v1/models 의 min/max/default_duration 노출
"""

from __future__ import annotations

import pytest

from app.backends import DurationOutOfRange, get_backend
from app.backends.sora import SoraBackend
from app.backends.veo import VeoBackend
from app.config import Settings

from .conftest import APP_KEY, MockBackend, auth_headers, wait_for_job


def _settings(**kw) -> Settings:
    return Settings(app_keys=APP_KEY, **kw)


# ---------------------------------------------------------------------- #
# 단위 테스트
# ---------------------------------------------------------------------- #
def test_validate_duration_rejects_out_of_range():
    b = SoraBackend(_settings())            # 지원: 4, 8, 12
    with pytest.raises(DurationOutOfRange):
        b.validate_duration(13.0)           # 최대(12) 초과
    with pytest.raises(DurationOutOfRange):
        b.validate_duration(2.0)            # 최소(4) 미만
    b.validate_duration(4.0)
    b.validate_duration(12.0)
    b.validate_duration(7.0)                # 범위 내 비지원 값은 통과(보정 대상)


def test_validate_duration_uses_registration_params():
    # 등록 파라미터(supported_durations)가 클래스 기본값보다 우선해야 한다.
    b = MockBackend(_settings(), supported_durations=(6.0, 20.0))
    b.validate_duration(20.0)
    with pytest.raises(DurationOutOfRange):
        b.validate_duration(4.0)


def test_normalize_duration_respects_instance_params():
    b = MockBackend(_settings(), supported_durations=(6.0, 20.0))
    assert b.normalize_duration(18.0) == 20.0
    assert b.normalize_duration(5.0) == 6.0


def test_default_duration_is_8_or_nearest():
    assert SoraBackend(_settings()).default_duration() == 8.0    # 4/8/12 → 8
    assert VeoBackend(_settings()).default_duration() == 8.0     # 4/6/8 → 8
    # 최대 5초 모델은 5초로 보정
    b = MockBackend(_settings(), supported_durations=(5.0,))
    assert b.default_duration() == 5.0


def test_registered_wan26_27_full_integer_range():
    # 공식 문서: wan2.6/2.7-t2v 는 2~15초 정수 (Alibaba Model Studio)
    s = _settings(dashscope_api_key="dummy")
    for name in ("wan-2.6", "wan-2.7"):
        b = get_backend(name, s)
        assert b.min_duration() == 2.0
        assert b.max_duration() == 15.0
        b.validate_duration(15.0)
        with pytest.raises(DurationOutOfRange):
            b.validate_duration(16.0)


# ---------------------------------------------------------------------- #
# API 테스트
# ---------------------------------------------------------------------- #
def test_message_rejects_duration_above_model_max(client):
    res = client.post(
        "/v1/videos/message",
        json={"prompt": "test ad", "model": "mock", "duration_sec": 10.0},
        headers=auth_headers(),
    )  # mock 지원: 2~4초 → 10초는 422
    assert res.status_code == 422
    assert "최대" in res.json()["detail"]


def test_message_accepts_duration_within_range(client):
    res = client.post(
        "/v1/videos/message",
        json={"prompt": "test ad", "model": "mock", "duration_sec": 4.0},
        headers=auth_headers(),
    )
    assert res.status_code == 202


def test_message_default_duration_normalized(client):
    # duration_sec 미지정 → 기본 8초를 mock 지원값(최대 4초)으로 보정해 사용
    res = client.post(
        "/v1/videos/message",
        json={"prompt": "test ad", "model": "mock"},
        headers=auth_headers(),
    )
    assert res.status_code == 202
    body = wait_for_job(client, res.json()["job_id"])
    assert body["status"] == "completed", body
    assert body["storyboard"]["scenes"][0]["duration_sec"] == 4.0


def test_models_expose_duration_limits(client):
    res = client.get("/v1/models", headers=auth_headers())
    models = {m["name"]: m for m in res.json()["models"]}
    sora = models["sora-2"]
    assert sora["min_duration"] == 4.0
    assert sora["max_duration"] == 12.0
    assert sora["default_duration"] == 8.0
    assert models["wan-2.6"]["max_duration"] == 15.0
    assert models["veo-3.1"]["max_duration"] == 8.0
    mock = models["mock"]
    assert mock["max_duration"] == 4.0
    assert mock["default_duration"] == 4.0
