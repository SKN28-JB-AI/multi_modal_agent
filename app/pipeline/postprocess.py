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


# 한글 자막 번인용 폰트 후보: (파일 경로, libass 가 인식하는 FontName).
# FontName 은 파일명이 아니라 폰트의 '패밀리 이름'이어야 한다.
_SUBTITLE_FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    (r"C:\Windows\Fonts\malgun.ttf", "Malgun Gothic"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic-Regular.ttf", "NanumGothic"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/Library/Fonts/AppleGothic.ttf", "AppleGothic"),
)


def _resolve_subtitle_font(font_path: str | None) -> tuple[str, str] | None:
    """
    번인 자막에 쓸 (fontsdir, FontName) 을 결정한다.

    우선순위:
      1) 명시된 font_path (TTF/TTC). FontName 은 후보표에서 찾고, 없으면
         파일 stem 을 사용.
      2) OS 별 잘 알려진 한글 폰트(_SUBTITLE_FONT_CANDIDATES).
    어느 것도 없으면 None(시스템 fontconfig 에 위임).
    """
    candidates: list[tuple[str, str]] = []
    if font_path and font_path.strip():
        fp = font_path.strip()
        known = dict(_SUBTITLE_FONT_CANDIDATES)
        family = known.get(fp) or Path(fp).stem
        candidates.append((fp, family))
    candidates.extend(_SUBTITLE_FONT_CANDIDATES)

    for fp, family in candidates:
        path = Path(fp)
        if path.is_file():
            return str(path.parent), family
    return None


def _escape_subtitles_path(path: Path) -> str:
    """subtitles/필터 인자용 경로 이스케이프(콜론/백슬래시/작은따옴표)."""
    return (
        str(path.resolve())
        .replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    )


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    font_path: str | None = None,
    font_size: int = 24,
) -> None:
    """
    SRT 자막을 영상에 굽는다(재인코딩).

    [글자 깨짐 방지]
    한글 폰트(FontName)와 fontsdir 을 명시해 libass 가 올바른 글리프를
    쓰게 한다. 폰트를 못 찾으면 시스템 fontconfig 에 위임하되, 가독성을
    위한 외곽선/그림자 스타일은 항상 적용한다.
    """
    ffmpeg = _ffmpeg()
    srt_escaped = _escape_subtitles_path(srt_path)

    # 가독성 스타일(외곽선+그림자) — 한글/배경 대비 확보
    style_parts = [
        f"Fontsize={font_size}",
        "PrimaryColour=&H00FFFFFF",   # 흰색 글자
        "OutlineColour=&H00000000",   # 검정 외곽선
        "BorderStyle=1",
        "Outline=2",
        "Shadow=1",
        "MarginV=40",
    ]

    resolved = _resolve_subtitle_font(font_path)
    sub_arg = f"subtitles='{srt_escaped}'"
    if resolved is not None:
        fontsdir, family = resolved
        style_parts.insert(0, f"FontName={family}")
        fontsdir_escaped = _escape_subtitles_path(Path(fontsdir))
        sub_arg += f":fontsdir='{fontsdir_escaped}'"

    sub_arg += ":force_style='" + ",".join(style_parts) + "'"

    _run(
        [
            ffmpeg, "-y", "-i", str(video_path),
            "-vf", sub_arg,
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
def _video_width(video_path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 1920
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return int(proc.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 1920


# 위치 → overlay 좌표식 매핑 (m = 여백 px)
_POSITIONS = {
    "top-right": ("W-w-{m}", "{m}"),
    "top-left": ("{m}", "{m}"),
    "bottom-right": ("W-w-{m}", "H-h-{m}"),
    "bottom-left": ("{m}", "H-h-{m}"),
}


def overlay_logo(
    video_path: Path,
    logo_path: Path,
    output_path: Path,
    scale_ratio: float = 0.12,
    opacity: float = 0.82,
    position: str = "top-right",
    margin_ratio: float = 0.03,
    fade_in_sec: float = 0.6,
) -> None:
    """
    로고를 방송 워터마크처럼 자연스럽게 오버레이한다(재인코딩 발생).

    - scale_ratio  : 로고 가로폭을 영상 가로폭의 비율로 자동 축소
    - opacity      : 반투명 처리(0~1)로 영상 위에 떠 보이지 않게
    - position     : top-right / top-left / bottom-right / bottom-left
    - margin_ratio : 가장자리 여백(영상 가로폭 비율)
    - fade_in_sec  : 시작 시 부드럽게 나타나는 페이드인
    """
    ffmpeg = _ffmpeg()
    vw = _video_width(video_path)
    logo_w = max(32, int(vw * scale_ratio))
    margin = max(8, int(vw * margin_ratio))
    opacity = min(max(opacity, 0.0), 1.0)

    x_expr, y_expr = _POSITIONS.get(position, _POSITIONS["top-right"])
    x = x_expr.format(m=margin)
    y = y_expr.format(m=margin)

    logo_chain = (
        f"[1:v]scale={logo_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={opacity:.2f}"
    )
    if fade_in_sec > 0:
        logo_chain += f",fade=t=in:st=0:d={fade_in_sec:.2f}:alpha=1"
    filter_complex = (
        f"{logo_chain}[lg];[0:v][lg]overlay={x}:{y}:shortest=1[v]"
    )

    _run(
        [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-loop", "1", "-i", str(logo_path),   # 페이드인을 위해 루프 입력
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            str(output_path),
        ],
        "로고 오버레이",
    )


# ---------------------------------------------------------------------- #
# 로고 아웃트로(엔드카드) — 광고 마지막에 로고 페이지를 붙인다.
# 후처리이므로 어떤 비디오 모델(Sora/Veo/LTX/Wan)로 만든 영상에도 동일 적용.
# ---------------------------------------------------------------------- #
def _video_dimensions(video_path: Path) -> tuple[int, int]:
    """영상의 (width, height)를 ffprobe 로 얻는다. 실패 시 1920x1080."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return (1920, 1080)
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        w, h = proc.stdout.strip().splitlines()[0].split("x")
        return (int(w), int(h))
    except (ValueError, IndexError):
        return (1920, 1080)


def _bg_to_ffmpeg_color(bg: str) -> str:
    """'#134A8E' / '134A8E' → ffmpeg color('0x134A8E'). 형식 오류 시 네이비."""
    import re

    v = (bg or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", v):
        v = "134A8E"
    return f"0x{v.upper()}"


def _video_framerate(video_path: Path) -> str:
    """
    참조 영상의 프레임레이트를 'num/den'(예: 30000/1001) 문자열로 얻는다.
    아웃트로를 본편과 같은 fps 로 만들어, concat 결과가 가변프레임(VFR)이 되어
    엔드카드가 더 빠르게(=짧게) 재생되는 문제를 막는다. 실패/이상값이면 '30'.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "30"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    raw = (proc.stdout or "").strip().splitlines()
    raw = raw[0].strip() if raw else ""
    try:
        if "/" in raw:
            num, den = raw.split("/", 1)
            val = float(num) / float(den) if float(den) else 0.0
        else:
            val = float(raw)
    except (ValueError, ZeroDivisionError):
        val = 0.0
    if not (1.0 <= val <= 240.0):
        return "30"
    return raw


def _audio_sample_rate(video_path: Path) -> int:
    """참조 영상 오디오의 샘플레이트(Hz). 없거나 이상값이면 48000."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 48000
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        sr = int((proc.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 48000
    return sr if 8000 <= sr <= 192000 else 48000


def make_logo_outro(
    logo_path: Path,
    reference_video: Path,
    out_path: Path,
    duration_sec: float = 2.5,
    bg_hex: str = "#134A8E",
    fade_sec: float = 0.4,
    scale_ratio: float = 0.42,
) -> None:
    """
    로고를 가운데 배치한 엔드카드 영상을 만든다.

    - reference_video 의 해상도/오디오 유무에 맞춰 생성하여, 본편과
      자연스럽게 이어 붙일(concat) 수 있게 한다.
    - bg_hex 배경 단색 위에 로고를 중앙 배치하고 페이드 인/아웃.
    - 본편에 오디오가 있으면 무음 스테레오 트랙을 넣어 스트림 구성을 맞춘다.
    """
    ffmpeg = _ffmpeg()
    w, h = _video_dimensions(reference_video)
    with_audio = _has_audio_stream(reference_video)
    # 본편과 동일한 fps/샘플레이트로 엔드카드를 만들어 CFR 결합을 보장한다.
    # (하드코딩 25fps 였을 때 Sora 등 다른 fps 본편과 concat 시 아웃트로가
    #  본편 fps 로 재생돼 2.5초보다 짧게 보이던 문제를 해결.)
    fps = _video_framerate(reference_video)
    sample_rate = _audio_sample_rate(reference_video)
    dur = max(0.5, float(duration_sec))
    fade = max(0.0, min(float(fade_sec), dur / 2))
    logo_w = max(48, int(w * max(0.05, min(scale_ratio, 0.95))))
    color = _bg_to_ffmpeg_color(bg_hex)

    inputs = [
        "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r={fps}:d={dur:.3f}",
        "-loop", "1", "-i", str(logo_path),
    ]
    # 로고: 프레임을 넘지 않도록 폭/높이 동시 제한 후 중앙 오버레이 + 페이드.
    vchain = (
        f"[1:v]scale={logo_w}:-1:force_original_aspect_ratio=decrease,"
        f"format=rgba[lg];"
        f"[0:v][lg]overlay=(W-w)/2:(H-h)/2:shortest=1"
    )
    if fade > 0:
        vchain += (
            f",fade=t=in:st=0:d={fade:.2f},"
            f"fade=t=out:st={max(0.0, dur - fade):.2f}:d={fade:.2f}"
        )
    vchain += "[v]"

    cmd = [ffmpeg, "-y", *inputs]
    maps = ["-map", "[v]"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}"]
        maps += ["-map", "2:a"]
    cmd += [
        "-filter_complex", vchain, *maps,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fps,
    ]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(out_path)]
    _run(cmd, "로고 아웃트로 생성")


def make_image_outro(
    image_path: Path,
    reference_video: Path,
    out_path: Path,
    duration_sec: float = 2.5,
    fade_sec: float = 0.4,
) -> None:
    """
    스타일화된 엔드카드 이미지(풀프레임)를 아웃트로 영상으로 만든다.

    make_logo_outro 와 동일하게 본편의 해상도/fps/오디오 유무에 맞춰
    생성해 concat 이 자연스럽도록 한다. 이미지는 프레임을 가득 채우도록
    cover(확대 후 중앙 crop) 처리한다.
    """
    ffmpeg = _ffmpeg()
    w, h = _video_dimensions(reference_video)
    with_audio = _has_audio_stream(reference_video)
    fps = _video_framerate(reference_video)
    sample_rate = _audio_sample_rate(reference_video)
    dur = max(0.5, float(duration_sec))
    fade = max(0.0, min(float(fade_sec), dur / 2))

    vchain = (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},format=yuv420p,fps={fps}"
    )
    if fade > 0:
        vchain += (
            f",fade=t=in:st=0:d={fade:.2f},"
            f"fade=t=out:st={max(0.0, dur - fade):.2f}:d={fade:.2f}"
        )
    vchain += "[v]"

    cmd = [ffmpeg, "-y", "-loop", "1", "-t", f"{dur:.3f}", "-i", str(image_path)]
    maps = ["-map", "[v]"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}"]
        maps += ["-map", "1:a"]
    cmd += [
        "-filter_complex", vchain, *maps,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fps,
    ]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(out_path)]
    _run(cmd, "이미지 아웃트로 생성")


def append_outro(main_video: Path, outro: Path, output_path: Path) -> None:
    """본편 뒤에 아웃트로를 이어 붙인다(concat, 필요 시 재인코딩)."""
    concat_clips([main_video, outro], output_path)
