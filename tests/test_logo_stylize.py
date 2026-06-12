"""
로고 아웃트로 스타일화(이미지 생성 API 기반 엔드카드) 테스트.

- stylize_logo_endcard: 비활성/키 없음 → None (폴백 신호)
- make_image_outro: 본편 해상도/오디오 구성에 맞는 유효 MP4 생성
- build_outro: 스타일화 성공 시 이미지 아웃트로, 실패 시 단색 폴백
- E2E: 스타일화 monkeypatch 후 메시지 잡에 아웃트로가 붙는지
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.llm.openai_llm import stylize_logo_endcard
from app.pipeline import outro as outro_mod
from app.pipeline.postprocess import make_image_outro

from .conftest import auth_headers, make_test_clip, wait_for_job
from .test_logo_outro import _dur, _logo


def _settings(**kw) -> Settings:
    return Settings(app_keys="test-key", _env_file=None, **kw)


def _probe(path: Path, stream: str, entry: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", f"stream={entry}",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out


# ---------------------------------------------------------------------- #
# stylize_logo_endcard 폴백 조건
# ---------------------------------------------------------------------- #
def test_stylize_returns_none_without_key(tmp_path):
    logo = tmp_path / "logo.png"; _logo(logo)
    res = asyncio.run(stylize_logo_endcard(
        _settings(openai_api_key=""), logo, "calm beach ad",
        tmp_path / "card.png",
    ))
    assert res is None


def test_stylize_returns_none_when_disabled(tmp_path):
    logo = tmp_path / "logo.png"; _logo(logo)
    res = asyncio.run(stylize_logo_endcard(
        _settings(openai_api_key="sk-x", logo_outro_stylize_enabled=False),
        logo, "calm beach ad", tmp_path / "card.png",
    ))
    assert res is None


def test_stylize_api_failure_falls_back_to_none(tmp_path, monkeypatch):
    # 키는 있으나 호출 실패(가짜 키 → SDK 예외) → None
    logo = tmp_path / "logo.png"; _logo(logo)

    import app.llm.openai_llm as m

    def boom():
        raise RuntimeError("api down")

    async def fake_to_thread(fn, *a, **kw):
        return boom()

    monkeypatch.setattr(m.asyncio, "to_thread", fake_to_thread)
    res = asyncio.run(stylize_logo_endcard(
        _settings(openai_api_key="sk-x"), logo, "ad", tmp_path / "card.png",
    ))
    assert res is None


# ---------------------------------------------------------------------- #
# make_image_outro
# ---------------------------------------------------------------------- #
def test_make_image_outro_matches_reference(tmp_path):
    ref = tmp_path / "ref.mp4"
    make_test_clip(ref, seconds=2.0, with_audio=True)
    card = tmp_path / "card.png"; _logo(card)   # 임의 PNG(풀프레임으로 cover됨)
    out = tmp_path / "outro.mp4"
    make_image_outro(card, ref, out, duration_sec=1.5, fade_sec=0.3)
    assert out.exists() and out.stat().st_size > 1000
    assert 1.2 < _dur(out) < 1.9
    # 본편과 동일 해상도 + 오디오 트랙 존재
    assert _probe(out, "v:0", "width") == _probe(ref, "v:0", "width")
    assert _probe(out, "a:0", "codec_type") == "audio"


def test_make_image_outro_no_audio(tmp_path):
    ref = tmp_path / "ref.mp4"
    make_test_clip(ref, seconds=1.0, with_audio=False)
    card = tmp_path / "card.png"; _logo(card)
    out = tmp_path / "outro.mp4"
    make_image_outro(card, ref, out, duration_sec=1.0)
    assert out.exists()
    assert _probe(out, "a", "codec_type") == ""   # 오디오 없음


# ---------------------------------------------------------------------- #
# build_outro 분기
# ---------------------------------------------------------------------- #
def test_build_outro_uses_stylized_card(tmp_path, monkeypatch):
    ref = tmp_path / "ref.mp4"; make_test_clip(ref, seconds=1.0)
    logo = tmp_path / "logo.png"; _logo(logo)

    async def fake_stylize(settings, logo_path, context, out_path, aspect_ratio="16:9"):
        _logo(out_path)           # 스타일화 결과 흉내
        return out_path

    monkeypatch.setattr(outro_mod, "stylize_logo_endcard", fake_stylize)
    out = asyncio.run(outro_mod.build_outro(
        _settings(logo_outro_duration_sec=1.0), logo, "ad", ref, tmp_path,
    ))
    assert out.exists()
    assert (tmp_path / "outro_card.png").exists()   # 스타일화 경로 사용 증거


def test_build_outro_falls_back_to_solid_bg(tmp_path, monkeypatch):
    ref = tmp_path / "ref.mp4"; make_test_clip(ref, seconds=1.0)
    logo = tmp_path / "logo.png"; _logo(logo)

    async def fake_stylize(*a, **kw):
        return None               # 스타일화 실패/비활성

    monkeypatch.setattr(outro_mod, "stylize_logo_endcard", fake_stylize)
    out = asyncio.run(outro_mod.build_outro(
        _settings(logo_outro_duration_sec=1.0), logo, "ad", ref, tmp_path,
    ))
    assert out.exists()
    assert not (tmp_path / "outro_card.png").exists()   # 폴백 경로


# ---------------------------------------------------------------------- #
# E2E: 메시지 잡 + 스타일화 아웃트로
# ---------------------------------------------------------------------- #
def test_message_outro_with_stylized_card(make_client, tmp_path, monkeypatch):
    logos = tmp_path / "logos"; logos.mkdir()
    _logo(logos / "default.png")

    async def fake_stylize(settings, logo_path, context, out_path, aspect_ratio="16:9"):
        assert "calm beach" in context        # 영상 프롬프트가 분위기 컨텍스트로 전달됨
        _logo(out_path)
        return out_path

    monkeypatch.setattr(outro_mod, "stylize_logo_endcard", fake_stylize)
    client = make_client(logos_dir=str(logos), logo_outro_duration_sec=2.0)
    res = client.post(
        "/v1/videos/message",
        json={"prompt": "a calm beach", "model": "mock",
              "duration_sec": 4.0, "logo_outro": True},
        headers=auth_headers(),
    )
    job_id = res.json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body
    video = client.get(body["video_url"], headers=auth_headers())
    out = tmp_path / "v.mp4"; out.write_bytes(video.content)
    assert _dur(out) > 3.5    # mock 2초 + 아웃트로 2초
