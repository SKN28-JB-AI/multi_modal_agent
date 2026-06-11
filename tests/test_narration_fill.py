"""
보이스오버/내레이션 분량을 '영상 길이에 맞게' 채우는 기능 검증.

대상:
  - app.narration : 언어별 발화 속도 → 클립 길이 기반 분량 산정(순수 함수)
  - 오케스트레이터 : 씬/단일 모드에서 백엔드 '실제' 클립 길이로 분량 산정(per-model)
  - 무음 모델(supports_audio=False) : 임베디드 보이스오버 생략
  - v2 build_video_prompt : 보이스오버 포함 + 반환값 존재(과거 절단 버그 회귀 방지)
  - GET /v1/models : supports_audio 노출
"""

from __future__ import annotations

import app.narration as nar
from app.backends.base import ClipResult, ClipSpec
from app.pipeline.orchestrator import Orchestrator
from app.schemas import Scene, Storyboard

from . import conftest as C
from .conftest import MockBackend, auth_headers, make_sample_pdf, wait_for_job


# ====================================================================== #
# 1) narration 모듈: 분량이 길이에 비례해 충분히 커진다
# ====================================================================== #
def test_budget_scales_with_duration_and_is_generous():
    b6 = nar.budget(6, "ko")
    b12 = nar.budget(12, "ko")
    # 과거 캡(6초=15자)보다 확실히 많다(꽉 채움).
    assert b6.target >= 24
    # 길이에 비례(2배 길면 대략 2배).
    assert b12.target > b6.target * 1.6
    assert b6.unit == "characters"


def test_budget_unit_by_language():
    assert nar.budget(8, "en").unit == "words"
    assert nar.budget(8, "ja").unit == "characters"
    # 알 수 없는 언어는 보수적으로 글자 기준.
    assert nar.budget(8, "xx").unit == "characters"


def test_language_normalization_aliases():
    assert nar.normalize_language("ko-KR") == "ko"
    assert nar.normalize_language("Japanese") == "ja"
    assert nar.normalize_language("") == "ko"


def test_voiceover_line_keeps_backcompat_substrings():
    line = nar.voiceover_line("시원한 하루", 6, "ko")
    assert 'says in Korean: "시원한 하루"' in line  # 3요소 호환
    assert "narrator" in line
    assert "6s" in line


def test_voiceover_multi_line_keeps_speaks_in():
    line = nar.voiceover_multi_line('"a" then "b"', 8, "ko")
    assert "speaks in Korean" in line
    assert '"a" then "b"' in line


# ====================================================================== #
# 2) 오케스트레이터: per-model(백엔드 보정 길이)로 분량 산정
# ====================================================================== #
def test_compose_prompt_uses_actual_clip_duration():
    """clip_duration 을 주면 그 길이 기준으로 분량을 잡는다(모델 보정값)."""
    scene = Scene(index=0, prompt="beach", duration_sec=6,
                  narration="시원한 하루를 시작하세요.")
    # 모델이 실제로 2초만 만든다면, 6초가 아니라 2초 기준 분량.
    p = Orchestrator._compose_prompt(scene, "ko", True, clip_duration=2.0)
    assert "entire 2s" in p
    assert 'says in Korean: "시원한 하루를 시작하세요."' in p


def test_scenes_mode_budget_matches_backend_duration(client, tmp_path):
    """E2E: mock(2초 고정) 모델 → 씬 프롬프트가 4초가 아닌 2초 기준으로 채워진다."""
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock-2s", "options": '{"generation_mode": "scenes"}'},
        headers=auth_headers(),
    )
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body
    prompts = MockBackend.captured_prompts
    assert len(prompts) == 2
    # 백엔드 보정 길이(2초)가 분량 산정에 쓰였는지
    assert all("entire 2s" in p for p in prompts)
    assert 'says in Korean: "시원한 하루를 시작하세요."' in prompts[0]


# ====================================================================== #
# 3) 무음 모델: 임베디드 보이스오버 생략
# ====================================================================== #
class SilentMockBackend(MockBackend):
    """오디오 미지원(무음) 모델 시뮬레이션. 클립 자체는 mock 과 동일."""

    description = "무음 mock 백엔드"
    supports_audio = False


def test_silent_backend_skips_embedded_voiceover(make_client, tmp_path):
    from app import backends
    backends.register("mock-silent", SilentMockBackend)
    try:
        client = make_client()
        pdf = tmp_path / "plan.pdf"
        make_sample_pdf(pdf)
        resp = client.post(
            "/v1/videos/pdf",
            files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
            data={"model": "mock-silent",
                  "options": '{"generation_mode": "scenes"}'},
            headers=auth_headers(),
        )
        body = wait_for_job(client, resp.json()["job_id"])
        assert body["status"] == "completed", body
        prompts = MockBackend.captured_prompts
        assert len(prompts) == 2
        # 무음 모델이므로 보이스오버 지시문이 빠져야 한다.
        assert all("Voiceover" not in p for p in prompts)
    finally:
        backends.unregister("mock-silent")


def test_silent_backend_audio_supported_flag():
    from app.config import Settings
    from app import backends
    backends.register("mock-silent2", SilentMockBackend)
    try:
        s = Settings(app_keys="x", _env_file=None)
        be = backends.get_backend("mock-silent2", s)
        assert be.audio_supported() is False
    finally:
        backends.unregister("mock-silent2")


# ====================================================================== #
# 4) v2 build_video_prompt: 보이스오버 포함 + 반환값(절단 버그 회귀 방지)
# ====================================================================== #
def test_build_video_prompt_returns_and_includes_voiceover():
    from app.ads.prompts import build_video_prompt
    from app.ads.schemas import AdStoryboard, Cut, Music

    sb = AdStoryboard(
        project="P", concept="C", target="T", mood=["warm"],
        total_duration_sec=6,
        music=Music(genre="jazz", bpm=90, key_moment="x"),
        cuts=[Cut(cut=1, timecode="00:00-00:06", duration_sec=6, title="t",
                  scene="a park", visual="walk", camera="dolly",
                  voiceover="새로운 금융을 만나보세요")],
    )
    p = build_video_prompt(sb.cuts[0], sb, locale="ko-KR")
    assert isinstance(p, str) and p          # 과거엔 None 을 반환했음
    assert 'says in Korean: "새로운 금융을 만나보세요"' in p
    # 무음 모델이면 보이스오버 생략
    p2 = build_video_prompt(sb.cuts[0], sb, locale="ko-KR", include_voiceover=False)
    assert "Voiceover" not in p2


# ====================================================================== #
# 5) /v1/models supports_audio 노출
# ====================================================================== #
def test_models_endpoint_exposes_audio_flag(client):
    models = {
        m["name"]: m
        for m in client.get("/v1/models", headers=auth_headers()).json()["models"]
    }
    assert models["wan-2.2"]["supports_audio"] is False
    assert models["wan-2.5"]["supports_audio"] is True
    assert models["sora-2"]["supports_audio"] is True
