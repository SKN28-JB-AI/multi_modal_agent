"""
메시지 모드(/v1/videos/message) 프롬프트 변환(선행 단계) 테스트.

검증:
  - 기본적으로 입력 프롬프트가 OpenAI 기본 모델로 변환되어 비디오 생성에 쓰인다
  - 요청 enhance_prompt=False 면 변환을 건너뛴다
  - 변환 실패 시 원본 프롬프트로 폴백하고 잡은 정상 완료된다
  - OPENAI_API_KEY 가 없으면 변환을 생략하고 원본을 쓴다
"""

from __future__ import annotations

import pytest

from app.pipeline import orchestrator as orch

from .conftest import auth_headers, wait_for_job


class EnhanceLLM:
    """변환 결과에 대상 모델명을 새겨 추적 가능하게 하는 가짜 LLM."""

    captured: list[dict] = []

    async def enhance_video_prompt(self, prompt, *, model, aspect_ratio="16:9",
                                   resolution="1080p", duration_sec=6.0,
                                   language="ko"):
        EnhanceLLM.captured.append(
            {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio,
             "resolution": resolution, "duration_sec": duration_sec}
        )
        return f"[ENHANCED:{model}] cinematic {prompt}"


class FailingEnhanceLLM:
    async def enhance_video_prompt(self, prompt, **kwargs):
        raise RuntimeError("의도된 변환 실패")


def _post(client, **extra):
    body = {"prompt": "a calm beach at sunset", "model": "mock",
            "duration_sec": 4.0}
    body.update(extra)
    return client.post("/v1/videos/message", json=body, headers=auth_headers())


def _scene0_prompt(body: dict) -> str:
    return body["storyboard"]["scenes"][0]["prompt"]


def test_prompt_enhanced_by_default(make_client, monkeypatch):
    EnhanceLLM.captured = []
    client = make_client()
    monkeypatch.setattr(orch, "get_llm", lambda settings: EnhanceLLM())

    resp = _post(client)
    assert resp.status_code == 202, resp.text
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body

    # 변환된 프롬프트가 스토리보드(=비디오 생성 입력)에 반영됐는지
    assert _scene0_prompt(body).startswith("[ENHANCED:mock]")
    # 변환 호출에 대상 모델/길이가 전달됐는지
    assert EnhanceLLM.captured and EnhanceLLM.captured[0]["model"] == "mock"
    assert EnhanceLLM.captured[0]["duration_sec"] == 4.0
    assert EnhanceLLM.captured[0]["prompt"] == "a calm beach at sunset"


def test_enhance_disabled_per_request(make_client, monkeypatch):
    EnhanceLLM.captured = []
    client = make_client()
    monkeypatch.setattr(orch, "get_llm", lambda settings: EnhanceLLM())

    resp = _post(client, enhance_prompt=False)
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed"
    # 변환을 건너뛰어 원본이 그대로 쓰인다
    assert _scene0_prompt(body) == "a calm beach at sunset"
    assert EnhanceLLM.captured == []


def test_enhance_disabled_by_server_default(make_client, monkeypatch):
    EnhanceLLM.captured = []
    client = make_client(enhance_message_prompt=False)
    monkeypatch.setattr(orch, "get_llm", lambda settings: EnhanceLLM())

    resp = _post(client)
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed"
    assert _scene0_prompt(body) == "a calm beach at sunset"
    assert EnhanceLLM.captured == []


def test_request_override_beats_server_default(make_client, monkeypatch):
    """서버 기본 off 라도 요청 enhance_prompt=True 면 변환한다."""
    EnhanceLLM.captured = []
    client = make_client(enhance_message_prompt=False)
    monkeypatch.setattr(orch, "get_llm", lambda settings: EnhanceLLM())

    resp = _post(client, enhance_prompt=True)
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed"
    assert _scene0_prompt(body).startswith("[ENHANCED:mock]")


def test_fallback_to_original_on_enhance_error(make_client, monkeypatch):
    client = make_client()
    monkeypatch.setattr(orch, "get_llm", lambda settings: FailingEnhanceLLM())

    resp = _post(client)
    body = wait_for_job(client, resp.json()["job_id"])
    # 변환 실패해도 잡은 원본으로 정상 완료
    assert body["status"] == "completed", body
    assert _scene0_prompt(body) == "a calm beach at sunset"


def test_enhance_skipped_without_openai_key(make_client, monkeypatch):
    EnhanceLLM.captured = []
    client = make_client(openai_api_key="")   # 키 없음 → 변환 생략
    monkeypatch.setattr(orch, "get_llm", lambda settings: EnhanceLLM())

    resp = _post(client)
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed"
    assert _scene0_prompt(body) == "a calm beach at sunset"
    assert EnhanceLLM.captured == []
