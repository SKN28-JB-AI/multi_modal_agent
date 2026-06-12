"""
글자 깨짐 방지 + 단계별 글자 노출(text-exposure) 가드레일 테스트.

- 단계 정규화 / 단계별 클로즈·negative 매핑
- Veo/LTX 가 단계별 negative_prompt 를, 세 백엔드가 단계별 긍정 클로즈를
  보내는지(네트워크 모킹으로 캡처). full 단계는 제약이 사라지는지.
- burn_subtitles 의 force_style(+한글 FontName/fontsdir)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.backends.base import (
    DEFAULT_TEXT_EXPOSURE,
    NO_TEXT_CLAUSE,
    NO_TEXT_NEGATIVE,
    ClipSpec,
    apply_text_policy,
    merge_negative_prompt,
    negative_for,
    negative_prompt_for,
    normalize_text_exposure,
    text_clause_for,
    with_no_text_clause,
)
from app.config import Settings
from app.pipeline import postprocess


def _settings(**kw):
    return Settings(app_keys="k", openai_api_key="x", gemini_api_key="y",
                    fal_api_key="z", _env_file=None, **kw)


# ---------------------------------------------------------------------- #
# 단계 정규화 / 매핑
# ---------------------------------------------------------------------- #
def test_default_exposure_is_minimal():
    assert DEFAULT_TEXT_EXPOSURE == "minimal"
    assert ClipSpec(prompt="x").text_exposure == "minimal"


def test_normalize_text_exposure():
    assert normalize_text_exposure(None) == "minimal"
    assert normalize_text_exposure("") == "minimal"
    assert normalize_text_exposure("FULL") == "full"
    assert normalize_text_exposure("none") == "none"
    assert normalize_text_exposure("nonsense") == "minimal"


def test_clause_levels_differ():
    # none 은 완전 금지, full 은 클로즈 없음
    assert "no on-screen text" in text_clause_for("none").lower()
    assert text_clause_for("full") == ""
    # minimal/moderate 는 한글 비라틴 억제 문구 포함
    assert "non-latin" in text_clause_for("minimal").lower()
    assert "non-latin" in text_clause_for("moderate").lower()


def test_negative_levels():
    assert negative_for("full") == ""
    assert "hangul text" in negative_for("none")
    assert "hangul text" in negative_for("minimal")
    assert "hangul text" in negative_for("moderate")
    # none 이 가장 광범위(자막/워터마크/UI 등 전부)
    assert "captions" in negative_for("none")


def test_apply_text_policy_full_is_noop():
    assert apply_text_policy("a beach", "full") == "a beach"


def test_apply_text_policy_minimal_appends_once():
    p = apply_text_policy("a beach", "minimal")
    assert "a beach" in p and "non-Latin" in p
    assert apply_text_policy(p, "minimal") == p   # 중복 방지


def test_backward_compat_helpers_map_to_none():
    assert with_no_text_clause("x") == apply_text_policy("x", "none")
    assert merge_negative_prompt(None) == NO_TEXT_NEGATIVE
    assert NO_TEXT_CLAUSE == text_clause_for("none")


def test_negative_prompt_for_merges_and_dedupes():
    merged = negative_prompt_for("none", "blurry")
    assert "blurry" in merged and "hangul text" in merged
    assert negative_prompt_for("none", merged) == merged


# ---------------------------------------------------------------------- #
# LTX payload 캡처
# ---------------------------------------------------------------------- #
def _run_ltx(monkeypatch, spec):
    import app.backends.ltx as ltx_mod
    captured = {}

    class _Resp:
        def __init__(self, d): self._d = d
        def raise_for_status(self): pass
        def json(self): return self._d

    class _FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None):
            captured["payload"] = json
            return _Resp({"status_url": "s", "response_url": "r"})
        async def get(self, url, headers=None, timeout=None):
            return _Resp({"status": "COMPLETED"} if url == "s"
                         else {"video": {"url": "http://v/x.mp4"}})
        def stream(self, *a, **k):
            raise RuntimeError("stop-before-download")

    monkeypatch.setattr(ltx_mod.httpx, "AsyncClient", _FakeClient)
    backend = ltx_mod.LtxBackend(_settings())
    import asyncio
    with pytest.raises(Exception):
        asyncio.run(
            backend.generate_clip(spec, Path("/tmp/_ltx_out.mp4")))
    return captured["payload"]


def test_ltx_minimal_sends_level_negative_and_clause(monkeypatch):
    payload = _run_ltx(monkeypatch, ClipSpec(prompt="seaside city"))  # 기본 minimal
    assert payload["negative_prompt"] == negative_for("minimal")
    assert "non-Latin" in payload["prompt"]


def test_ltx_full_drops_negative_and_clause(monkeypatch):
    payload = _run_ltx(
        monkeypatch, ClipSpec(prompt="seaside city", text_exposure="full"))
    assert payload["negative_prompt"] == ""        # full → 제약 없음
    assert payload["prompt"] == "seaside city"     # 클로즈 미부착


# ---------------------------------------------------------------------- #
# Veo config 캡처
# ---------------------------------------------------------------------- #
def _run_veo(monkeypatch, spec):
    import sys
    import types as _t
    captured = {}

    fake_genai = _t.ModuleType("google.genai")
    fake_types = _t.ModuleType("google.genai.types")

    class _Cfg:
        def __init__(self, **kw): captured["config"] = kw

    class _Models:
        def generate_videos(self, model, prompt, config, **kw):
            captured["prompt"] = prompt
            raise RuntimeError("stop-after-submit")

    class _Client:
        def __init__(self, *a, **k): self.models = _Models()

    fake_types.GenerateVideosConfig = _Cfg
    fake_types.Image = _t.SimpleNamespace(from_file=lambda location: object())
    fake_genai.Client = _Client
    fake_google = _t.ModuleType("google"); fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    import app.backends.veo as veo_mod
    backend = veo_mod.VeoBackend(_settings())
    import asyncio
    with pytest.raises(Exception):
        asyncio.run(
            backend.generate_clip(spec, Path("/tmp/_veo_out.mp4")))
    return captured


def test_veo_minimal_sends_level_negative_and_clause(monkeypatch):
    cap = _run_veo(monkeypatch, ClipSpec(prompt="seaside city"))
    assert cap["config"]["negative_prompt"] == negative_prompt_for("minimal")
    assert "non-Latin" in cap["prompt"]


def test_veo_full_has_no_negative(monkeypatch):
    cap = _run_veo(monkeypatch, ClipSpec(prompt="seaside city",
                                         text_exposure="full"))
    assert "negative_prompt" not in cap["config"]   # full → 미설정
    assert cap["prompt"] == "seaside city"


# ---------------------------------------------------------------------- #
# Sora prompt 캡처
# ---------------------------------------------------------------------- #
def _run_sora(monkeypatch, spec):
    import app.backends.sora as sora_mod
    captured = {}

    class _Videos:
        def create(self, **kw):
            captured.update(kw)
            raise RuntimeError("stop-after-submit")

    class _Client:
        def __init__(self, *a, **k): self.videos = _Videos()

    monkeypatch.setattr(sora_mod.SoraBackend, "_client", lambda self: _Client())
    backend = sora_mod.SoraBackend(_settings(), model="sora-2")
    import asyncio
    with pytest.raises(Exception):
        asyncio.run(
            backend.generate_clip(spec, Path("/tmp/_sora_out.mp4")))
    return captured


def test_sora_none_appends_full_ban(monkeypatch):
    cap = _run_sora(monkeypatch,
                    ClipSpec(prompt="seaside city", resolution="720p",
                             text_exposure="none"))
    assert NO_TEXT_CLAUSE in cap["prompt"]


def test_sora_full_leaves_prompt_unchanged(monkeypatch):
    cap = _run_sora(monkeypatch,
                    ClipSpec(prompt="seaside city", resolution="720p",
                             text_exposure="full"))
    assert cap["prompt"] == "seaside city"


# ---------------------------------------------------------------------- #
# 자막 번인: force_style + 한글 폰트
# ---------------------------------------------------------------------- #
def test_burn_subtitles_applies_force_style(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(postprocess, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(postprocess, "_run",
                        lambda cmd, what: captured.update(cmd=cmd))
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\n안녕\n", encoding="utf-8")
    postprocess.burn_subtitles(tmp_path / "in.mp4", srt, tmp_path / "out.mp4")
    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert "force_style=" in vf and "Outline=" in vf and "MarginV=" in vf


def test_burn_subtitles_uses_explicit_font(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(postprocess, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(postprocess, "_run",
                        lambda cmd, what: captured.update(cmd=cmd))
    font = tmp_path / "MyKR.ttf"
    font.write_bytes(b"\x00\x01\x00\x00fake-ttf")
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n가\n", encoding="utf-8")
    postprocess.burn_subtitles(
        tmp_path / "in.mp4", srt, tmp_path / "out.mp4", font_path=str(font))
    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert "FontName=MyKR" in vf and "fontsdir=" in vf
