"""④ 후처리(FFmpeg) 단위 테스트."""

import subprocess

import pytest

from app.pipeline import postprocess
from app.schemas import Scene, Storyboard

from .conftest import make_test_clip


def _duration_of(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_concat_two_clips(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make_test_clip(a, 2.0, "red")
    make_test_clip(b, 2.0, "blue")
    merged = tmp_path / "merged.mp4"
    postprocess.concat_clips([a, b], merged)
    assert merged.exists()
    assert _duration_of(merged) == pytest.approx(4.0, abs=0.5)


def test_concat_single_clip_copies(tmp_path):
    a = tmp_path / "a.mp4"
    make_test_clip(a, 2.0)
    out = tmp_path / "out.mp4"
    postprocess.concat_clips([a], out)
    assert out.exists() and out.stat().st_size == a.stat().st_size


def test_concat_empty_raises(tmp_path):
    with pytest.raises(postprocess.PostprocessError):
        postprocess.concat_clips([], tmp_path / "x.mp4")


def test_write_srt_timing(tmp_path):
    sb = Storyboard(
        title="t",
        scenes=[
            Scene(index=0, prompt="p1", on_screen_text="첫 번째"),
            Scene(index=1, prompt="p2", on_screen_text=""),       # 자막 없음
            Scene(index=2, prompt="p3", on_screen_text="세 번째"),
        ],
    )
    srt = tmp_path / "s.srt"
    assert postprocess.write_srt(sb, [4.0, 6.0, 8.0], srt) is True
    text = srt.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:04,000" in text   # 씬1
    assert "00:00:10,000 --> 00:00:18,000" in text   # 씬3 (4+6 초 후)
    assert "두 번째" not in text


def test_write_srt_no_text_returns_false(tmp_path):
    sb = Storyboard(title="t", scenes=[Scene(index=0, prompt="p")])
    assert postprocess.write_srt(sb, [4.0], tmp_path / "s.srt") is False
    assert not (tmp_path / "s.srt").exists()


def test_mix_narration_onto_video(tmp_path):
    video = tmp_path / "v.mp4"
    make_test_clip(video, 2.0, with_audio=True)
    narration = tmp_path / "n.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=5",
         str(narration)],
        check=True, capture_output=True,
    )
    out = tmp_path / "mixed.mp4"
    postprocess.mix_narration(video, narration, out)
    assert out.exists()
    # duration=first → 영상 길이(2초)에 맞춰 잘려야 함
    assert _duration_of(out) == pytest.approx(2.0, abs=0.5)


def test_overlay_logo(tmp_path):
    video = tmp_path / "v.mp4"
    make_test_clip(video, 2.0)
    logo = tmp_path / "logo.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=64x64:d=1",
         "-frames:v", "1", str(logo)],
        check=True, capture_output=True,
    )
    out = tmp_path / "branded.mp4"
    postprocess.overlay_logo(video, logo, out)
    assert out.exists() and out.stat().st_size > 0


def test_overlay_logo_scaled_and_translucent(tmp_path):
    """로고가 영상 폭 비율로 축소되고 반투명·여백이 적용되는지 픽셀 검증."""
    video = tmp_path / "v.mp4"
    make_test_clip(video, 2.0, "black")            # 검은 배경 320x180
    logo = tmp_path / "logo.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=200x200:d=1",
         "-frames:v", "1", str(logo)],
        check=True, capture_output=True,
    )
    out = tmp_path / "branded.mp4"
    postprocess.overlay_logo(
        video, logo, out,
        scale_ratio=0.2, opacity=0.5, position="top-right",
        margin_ratio=0.05, fade_in_sec=0.0,
    )
    # 마지막 프레임 추출 후 픽셀 검사
    frame = tmp_path / "frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(out),
         "-frames:v", "1", str(frame)],
        check=True, capture_output=True,
    )
    try:
        from PIL import Image
    except ImportError:
        import pytest
        pytest.skip("Pillow 없음")

    img = Image.open(frame).convert("RGB")
    w, h = img.size                      # 320x180
    margin = int(w * 0.05)               # 16px
    logo_w = int(w * 0.2)                # 64px
    # 로고 중앙부(우상단): 반투명 흰색 → 회색(~128). 0.5 투명도 검증
    cx = w - margin - logo_w // 2
    cy = margin + logo_w // 2
    r, g, b = img.getpixel((cx, cy))
    assert 90 < r < 180, f"로고 영역 밝기 이상(불투명 적용 안 됨?): {r}"
    # 영상 중앙(로고 밖)은 그대로 검정
    r2, g2, b2 = img.getpixel((w // 2, h - 10))
    assert r2 < 30, f"로고 밖 영역 오염: {r2}"
