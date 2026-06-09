"""
로고 아웃트로(엔드카드) 기능 테스트.

- make_logo_outro: 본편 해상도/오디오 유무에 맞춰 유효 MP4 생성 + concat
- _bg_to_ffmpeg_color: 색 형식 정규화/폴백
- recommend_outro_background: 키 없을 때 폴백
- /v1/videos/message: logo_outro 미지정→미적용, true→최종 영상 길이 증가
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from .conftest import APP_KEY, auth_headers, wait_for_job

from app.config import Settings
from app.llm import openai_llm
from app.pipeline.postprocess import (
    _bg_to_ffmpeg_color,
    _has_audio_stream,
    append_outro,
    make_logo_outro,
)


def _dur(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(p)],
        capture_output=True, text=True)
    return float(r.stdout.strip())


def _make_ref(path: Path, color="red", with_audio=True, dur=2.0):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={dur}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _logo(path: Path):
    # 작은 PNG 로고 생성(ffmpeg)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=120x60:d=1",
         "-frames:v", "1", str(path)], check=True, capture_output=True)


# ---------------------------------------------------------------------- #
def test_bg_color_normalization():
    assert _bg_to_ffmpeg_color("#134A8E") == "0x134A8E"
    assert _bg_to_ffmpeg_color("134a8e") == "0x134A8E"
    assert _bg_to_ffmpeg_color("nope") == "0x134A8E"   # 폴백
    assert _bg_to_ffmpeg_color("") == "0x134A8E"


def test_make_outro_matches_audio_present(tmp_path):
    ref = tmp_path / "ref.mp4"; _make_ref(ref, with_audio=True)
    logo = tmp_path / "logo.png"; _logo(logo)
    outro = tmp_path / "outro.mp4"
    make_logo_outro(logo, ref, outro, duration_sec=1.5, bg_hex="#134A8E")
    assert outro.exists() and _has_audio_stream(outro)
    assert abs(_dur(outro) - 1.5) < 0.4
    final = tmp_path / "final.mp4"
    append_outro(ref, outro, final)
    assert _dur(final) > _dur(ref)   # 길이 증가


def test_make_outro_matches_no_audio(tmp_path):
    ref = tmp_path / "ref.mp4"; _make_ref(ref, with_audio=False)
    logo = tmp_path / "logo.png"; _logo(logo)
    outro = tmp_path / "outro.mp4"
    make_logo_outro(logo, ref, outro, duration_sec=1.0, bg_hex="white")
    assert outro.exists() and not _has_audio_stream(outro)


def test_recommend_background_fallback_without_key():
    s = Settings(app_keys="k", openai_api_key="", _env_file=None)
    r = asyncio.get_event_loop().run_until_complete(
        openai_llm.recommend_outro_background(s, "beach ad", "JB"))
    assert r == "#134A8E"


# ---------------------------------------------------------------------- #
# message 모드 E2E (mock 백엔드 + outro)
# ---------------------------------------------------------------------- #
def _post(client, **extra):
    body = {"prompt": "a calm beach", "model": "mock", "duration_sec": 4.0}
    body.update(extra)
    return client.post("/v1/videos/message", json=body, headers=auth_headers())


def test_message_no_outro_by_default(make_client, tmp_path):
    # logos 폴더에 로고를 둬도 기본은 미적용(opt-in)
    logos = tmp_path / "logos"; logos.mkdir()
    _logo(logos / "default.png")
    client = make_client(logos_dir=str(logos))
    job_id = _post(client).json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed"
    # 최종 길이가 클립(2초 mock)과 유사 — 아웃트로 미추가
    video = client.get(body["video_url"], headers=auth_headers())
    out = tmp_path / "v.mp4"; out.write_bytes(video.content)
    assert _dur(out) < 3.5


def test_message_with_outro_appends_endcard(make_client, tmp_path):
    logos = tmp_path / "logos"; logos.mkdir()
    _logo(logos / "default.png")
    client = make_client(logos_dir=str(logos), logo_outro_duration_sec=2.0)
    job_id = _post(client, logo_outro=True).json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body
    video = client.get(body["video_url"], headers=auth_headers())
    out = tmp_path / "v.mp4"; out.write_bytes(video.content)
    # mock 클립 2초 + 아웃트로 2초 ≈ 4초
    assert _dur(out) > 3.5


def test_message_outro_skipped_when_no_logo(make_client, tmp_path):
    logos = tmp_path / "empty_logos"; logos.mkdir()   # 로고 없음
    client = make_client(logos_dir=str(logos))
    job_id = _post(client, logo_outro=True).json()["job_id"]
    body = wait_for_job(client, job_id)
    # 로고가 없으면 아웃트로 생략하되 잡은 정상 완료
    assert body["status"] == "completed"
    video = client.get(body["video_url"], headers=auth_headers())
    out = tmp_path / "v.mp4"; out.write_bytes(video.content)
    assert _dur(out) < 3.5


def _probe_fps(path: Path) -> tuple[str, str]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,avg_frame_rate",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    return (lines[0].strip(), lines[1].strip()) if len(lines) >= 2 else ("", "")


def _make_ref_fps(path: Path, fps: str, dur=3.0):
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c=green:s=640x360:r={fps}:d={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fps,
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


def test_outro_matches_reference_fps_and_concat_is_cfr(tmp_path):
    """
    회귀: 아웃트로가 본편과 다른 fps(과거 하드코딩 25fps)로 만들어지면 concat 결과가
    VFR 이 되어 엔드카드가 본편 fps 로 재생→짧게 보였다. 이제 본편 fps 를 따라가
    결합본이 CFR 이고 아웃트로가 요청 길이만큼 온전히 재생되어야 한다.
    """
    for fps in ("30", "24", "60"):
        ref = tmp_path / f"ref_{fps.replace('/', '_')}.mp4"
        _make_ref_fps(ref, fps)
        logo = tmp_path / "logo.png"; _logo(logo)
        outro = tmp_path / f"outro_{fps}.mp4"
        make_logo_outro(logo, ref, outro, duration_sec=2.5, bg_hex="#134A8E")

        # 아웃트로가 본편과 같은 fps 로 생성됐는지(ffprobe 는 'num/den' 표기)
        assert _probe_fps(outro)[0] == _probe_fps(ref)[0], (
            fps, _probe_fps(outro), _probe_fps(ref))

        final = tmp_path / f"final_{fps}.mp4"
        append_outro(ref, outro, final)
        # 결합본이 CFR(r_frame_rate == avg_frame_rate)
        r_fps, avg_fps = _probe_fps(final)
        assert r_fps == avg_fps, (fps, r_fps, avg_fps)
        # 아웃트로 분량이 요청 길이(2.5s)에 근접(본편 3s + 2.5s)
        outro_portion = _dur(final) - _dur(ref)
        assert abs(outro_portion - 2.5) < 0.2, (fps, outro_portion)
