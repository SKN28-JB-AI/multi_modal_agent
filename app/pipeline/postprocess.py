"""
pipeline/postprocess.py
-----------------------
④ 단계: FFmpeg 기반 후처리.

- concat_clips    : 씬 클립들을 하나의 MP4 로 결합(무손실 시도 후 재인코딩 폴백)
- write_srt       : 스토리보드의 on_screen_text → SRT 자막 파일
- burn_subtitles  : SRT 를 영상에 굽기(선택, 재인코딩 발생)
- synthesize_narration / mix_narration : TTS 내레이션 생성·합성(선택)

모든 함수는 동기(subprocess)이며 orchestrator 가 to_thread 로 호출한다.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..schemas import Storyboard


class PostprocessError(Exception):
    """후처리 실패."""


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise PostprocessError(
            "FFmpeg 가 설치되어 있지 않습니다. 후처리에 FFmpeg 가 필요합니다."
        )
    return path


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostprocessError(f"{what} 실패:\n{proc.stderr[-2000:]}")


# ---------------------------------------------------------------------- #
# 클립 결합
# ---------------------------------------------------------------------- #
def concat_clips(clip_paths: list[Path], output_path: Path) -> None:
    """클립들을 하나의 MP4 로 결합. -c copy 시도 후 실패하면 재인코딩."""
    if not clip_paths:
        raise PostprocessError("결합할 클립이 없습니다.")
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], output_path)
        return

    ffmpeg = _ffmpeg()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        list_file = Path(tf.name)
        for clip in clip_paths:
            safe = str(clip.resolve()).replace("'", r"'\''")
            tf.write(f"file '{safe}'\n")

    try:
        copy_cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(output_path),
        ]
        proc = subprocess.run(copy_cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        reencode_cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        _run(reencode_cmd, "FFmpeg 클립 결합(재인코딩)")
    finally:
        list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------- #
# 자막
# ---------------------------------------------------------------------- #
def _fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(storyboard: Storyboard, actual_durations: list[float],
              output_path: Path) -> bool:
    """
    씬별 on_screen_text 를 SRT 로 출력한다.
    actual_durations: 백엔드 보정 후의 실제 클립 길이(초) 목록.
    텍스트가 하나도 없으면 파일을 만들지 않고 False 를 반환.
    """
    entries: list[str] = []
    cursor = 0.0
    seq = 1
    for scene, dur in zip(storyboard.scenes, actual_durations):
        text = scene.on_screen_text.strip()
        if text:
            start, end = cursor, cursor + dur
            entries.append(f"{seq}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text}\n")
            seq += 1
        cursor += dur
    if not entries:
        return False
    output_path.write_text("\n".join(entries), encoding="utf-8")
    return True


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    """SRT 자막을 영상에 굽는다(재인코딩). 폰트 환경에 따라 결과가 달라짐."""
    ffmpeg = _ffmpeg()
    # subtitles 필터의 경로 이스케이프(콜론/백슬래시/작은따옴표)
    srt_escaped = (
        str(srt_path.resolve())
        .replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    )
    _run(
        [
            ffmpeg, "-y", "-i", str(video_path),
            "-vf", f"subtitles='{srt_escaped}'",
            "-c:a", "copy", str(output_path),
        ],
        "자막 번인",
    )


# ---------------------------------------------------------------------- #
# 내레이션 (선택)
# ---------------------------------------------------------------------- #
def synthesize_narration(
    script: str, api_key: str, model: str, voice: str, output_path: Path
) -> None:
    """OpenAI TTS 로 내레이션 오디오(mp3)를 생성한다."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=script
    ) as resp:
        resp.stream_to_file(str(output_path))


def _has_audio_stream(video_path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True  # 판별 불가 시 있다고 가정(amix 경로)
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def mix_narration(video_path: Path, narration_path: Path,
                  output_path: Path) -> None:
    """
    내레이션을 영상 오디오와 합성한다.
    - 원본에 오디오가 있으면: 원본 볼륨을 낮추고(0.35) 내레이션과 amix.
    - 없으면: 내레이션을 오디오 트랙으로 추가.
    영상 길이에 맞춰 잘린다(duration=first).
    """
    ffmpeg = _ffmpeg()
    if _has_audio_stream(video_path):
        filter_complex = (
            "[0:a]volume=0.35[bg];"
            "[bg][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        cmd = [
            ffmpeg, "-y", "-i", str(video_path), "-i", str(narration_path),
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg, "-y", "-i", str(video_path), "-i", str(narration_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_path),
        ]
    _run(cmd, "내레이션 합성")


# ---------------------------------------------------------------------- #
# 로고 오버레이 (선택)
# ---------------------------------------------------------------------- #
def overlay_logo(video_path: Path, logo_path: Path, output_path: Path,
                 margin: int = 40) -> None:
    """로고 PNG 를 우상단에 오버레이한다(재인코딩 발생)."""
    ffmpeg = _ffmpeg()
    _run(
        [
            ffmpeg, "-y", "-i", str(video_path), "-i", str(logo_path),
            "-filter_complex", f"[0:v][1:v]overlay=W-w-{margin}:{margin}[v]",
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            str(output_path),
        ],
        "로고 오버레이",
    )
