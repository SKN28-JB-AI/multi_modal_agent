"""
ads/pdf_gen.py
--------------
[4단계] 스토리보드 기반 광고 기획서 PDF 생성.

- ReportLab Platypus(SimpleDocTemplate) 사용.
- 한글 폰트는 ①설정/환경변수 TTF ②OS 별 잘 알려진 한글 TTF
  ③ReportLab 내장 한국어 CID 폰트(HYGothic-Medium) 순으로 등록한다.
- 2/3단계와 무관하게 실행 가능하되(요구사항 4), 컷 이미지가 이미
  생성되어 있으면 기획서에 함께 삽입해 완성도를 높인다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import Settings
from .schemas import AdStoryboard


class PdfGenerationError(Exception):
    """PDF 생성 실패."""


# 브랜드 컬러(JB금융그룹 블루 계열).
_BRAND_HEX = "#134A8E"
_LIGHT_HEX = "#EAF1FA"

# TTF 임베드 후보(위에서부터 탐색).
_TTF_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\malgun.ttf",                       # Windows 맑은 고딕
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",    # Linux 나눔고딕
    "/usr/share/fonts/truetype/nanum/NanumGothic-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/Library/Fonts/AppleGothic.ttf",                     # macOS
)

# 폴백: ReportLab 내장 한국어 CID 폰트(뷰어 의존적이지만 항상 동작).
_CID_FALLBACK = "HYGothic-Medium"


def generate_proposal_pdf(
    settings: Settings,
    storyboard: AdStoryboard,
    out_path: Path,
    cut_image_paths: Optional[dict[int, Path]] = None,
    source_prompt: str = "",
) -> Path:
    """
    스토리보드로 광고 기획서 PDF 를 만든다(동기 — to_thread 로 호출할 것).

    Parameters
    ----------
    cut_image_paths : 이미 생성된 컷 이미지 {컷 번호: 경로}. 없으면 텍스트만.
    source_prompt   : 기획의 출발점이 된 사용자 프롬프트(부록에 기재).
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Image as RLImage,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise PdfGenerationError(
            f"reportlab 패키지가 없습니다: {exc}. "
            "pip install reportlab 후 다시 시도하세요."
        ) from exc

    brand = colors.HexColor(_BRAND_HEX)
    light = colors.HexColor(_LIGHT_HEX)

    # ------------------------------------------------------------------ #
    # 한글 폰트 등록
    # ------------------------------------------------------------------ #
    def _register_font() -> str:
        candidates: list[str] = []
        if settings.ad_pdf_font_path.strip():
            candidates.append(settings.ad_pdf_font_path.strip())
        candidates.extend(_TTF_CANDIDATES)
        for ttf_path in candidates:
            if Path(ttf_path).is_file():
                try:
                    pdfmetrics.registerFont(TTFont("KRFont", ttf_path))
                    return "KRFont"
                except Exception:  # noqa: BLE001 - 다음 후보로
                    continue
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(_CID_FALLBACK))
            return _CID_FALLBACK
        except Exception as exc:  # noqa: BLE001
            raise PdfGenerationError(f"한글 폰트 등록 실패: {exc}") from exc

    font_name = _register_font()
    base = dict(fontName=font_name, wordWrap="CJK")
    st = {
        "title": ParagraphStyle(
            "title", fontSize=24, leading=32, textColor=brand, spaceAfter=6, **base
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=13, leading=20,
            textColor=colors.HexColor("#444444"), spaceAfter=4, **base,
        ),
        "h2": ParagraphStyle(
            "h2", fontSize=15, leading=22, textColor=brand,
            spaceBefore=14, spaceAfter=6, **base,
        ),
        "body": ParagraphStyle("body", fontSize=10, leading=16, **base),
        "small": ParagraphStyle(
            "small", fontSize=8.5, leading=13,
            textColor=colors.HexColor("#666666"), **base,
        ),
        "cell": ParagraphStyle("cell", fontSize=9, leading=14, **base),
        "cell_label": ParagraphStyle(
            "cell_label", fontSize=9, leading=14, textColor=brand, **base
        ),
    }

    def _esc(text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _kv_table(rows: list[list[str]]):
        data = [
            [Paragraph(_esc(k), st["cell_label"]), Paragraph(_esc(v), st["cell"])]
            for k, v in rows
        ]
        table = Table(data, colWidths=[32 * mm, 138 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), light),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D7EA")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def _scaled_image(path: Path, max_width: float):
        from PIL import Image as PILImage

        with PILImage.open(path) as img:
            w, h = img.size
        if w <= 0:
            raise PdfGenerationError(f"이미지 크기 오류: {path}")
        scale = max_width / w
        return RLImage(str(path), width=max_width, height=h * scale)

    # ------------------------------------------------------------------ #
    # 문서 구성
    # ------------------------------------------------------------------ #
    sb = storyboard
    images = cut_image_paths or {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        title=f"{sb.project} 광고 기획서",
        author=sb.logo or "JB금융그룹",
    )

    story: list = []

    # 표지/헤더
    story.append(Paragraph(_esc(sb.project), st["title"]))
    story.append(
        Paragraph(f"광고 캠페인 기획서 · {_esc(sb.logo)}", st["subtitle"])
    )
    story.append(Paragraph(datetime.now().strftime("작성일 %Y-%m-%d"), st["small"]))
    story.append(Spacer(1, 6 * mm))

    # 1. 캠페인 개요
    story.append(Paragraph("1. 캠페인 개요", st["h2"]))
    story.append(
        _kv_table(
            [
                ["컨셉", sb.concept],
                ["타겟", sb.target],
                ["무드", " · ".join(sb.mood)],
                ["전체 길이", f"{sb.total_duration_sec}초 ({len(sb.cuts)}컷)"],
                ["화면 비율 / 포맷", f"{sb.aspect_ratio} / {sb.format or '-'}"],
                ["CTA", sb.cta or "-"],
            ]
        )
    )

    # 2. 음악 연출
    story.append(Paragraph("2. 음악 연출", st["h2"]))
    story.append(
        _kv_table(
            [
                ["장르", sb.music.genre],
                ["BPM", str(sb.music.bpm)],
                ["키 모먼트", sb.music.key_moment],
            ]
        )
    )

    # 3. 컷별 구성
    story.append(Paragraph("3. 컷별 구성", st["h2"]))
    for cut in sb.cuts:
        story.append(Spacer(1, 3 * mm))
        header = (
            f"CUT {cut.cut} · {_esc(cut.title)} "
            f"({_esc(cut.timecode)}, {cut.duration_sec}초)"
        )
        story.append(Paragraph(header, st["cell_label"]))
        story.append(Spacer(1, 1.5 * mm))
        story.append(
            _kv_table(
                [
                    ["장면", cut.scene],
                    ["비주얼", cut.visual],
                    ["카메라", cut.camera],
                    ["자막", cut.on_screen_text or "-"],
                    ["보이스오버", cut.voiceover or "-"],
                    ["사운드", cut.sfx or "-"],
                    ["전환", cut.transition or "-"],
                ]
            )
        )

        # 생성된 첫 장면 이미지가 있으면 삽입(없어도 무방 — 요구사항 4).
        img_path = images.get(cut.cut)
        if img_path and Path(img_path).exists():
            try:
                story.append(Spacer(1, 2 * mm))
                story.append(_scaled_image(Path(img_path), max_width=120 * mm))
                story.append(
                    Paragraph(
                        f"▲ CUT {cut.cut} 첫 장면(AI 생성 시안)", st["small"]
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 이미지는 비필수
                story.append(
                    Paragraph(
                        f"(컷 {cut.cut} 이미지 삽입 실패: {_esc(str(exc))})",
                        st["small"],
                    )
                )

    # 4. 제작 노트
    story.append(PageBreak())
    story.append(Paragraph("4. 제작 노트", st["h2"]))
    notes = [
        "본 기획서는 AI 파이프라인(스토리보드 → 컷 이미지 → 비디오 생성)으로 "
        "제작된 초안이며, 집행 전 브랜드/법무 검수가 필요합니다.",
        "금융 광고 심의 기준에 따라 수익률·혜택 표현의 사실 여부를 확인하십시오.",
        f"영상 생성 규격: {sb.aspect_ratio}, 총 {sb.total_duration_sec}초, "
        f"{len(sb.cuts)}컷 구성.",
    ]
    for n in notes:
        story.append(Paragraph("· " + _esc(n), st["body"]))
        story.append(Spacer(1, 1.5 * mm))

    if source_prompt:
        story.append(Paragraph("부록. 원본 프롬프트", st["h2"]))
        story.append(Paragraph(_esc(source_prompt), st["body"]))

    try:
        doc.build(story)
    except Exception as exc:  # noqa: BLE001
        raise PdfGenerationError(f"PDF 빌드 실패: {exc}") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise PdfGenerationError("PDF 파일이 생성되지 않았습니다.")
    return out_path
