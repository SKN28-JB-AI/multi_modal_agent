"""
pipeline/pdf_parser.py
----------------------
① 단계: PDF 기획서에서 텍스트 + 페이지 이미지를 추출한다 (PyMuPDF).

추출물은 비전 LLM(analyze_pdf)에 전달되어 광고 브리프로 요약된다.
텍스트만 뽑지 않고 페이지를 통째로 이미지화하는 이유:
기획서의 레이아웃·무드보드·제품컷·도표가 텍스트 추출에선 사라지기 때문.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class PdfParseError(Exception):
    """PDF 파싱 실패."""


@dataclass
class ParsedPdf:
    text: str
    page_images: list[bytes] = field(default_factory=list)  # PNG bytes
    page_count: int = 0
    truncated: bool = False  # max_pages 로 잘렸는지


def extract_pdf(path: Path, max_pages: int = 10, dpi: int = 110) -> ParsedPdf:
    """PDF 에서 텍스트와 페이지 PNG 이미지를 추출한다. (동기 — to_thread 로 호출)"""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise PdfParseError(f"PyMuPDF 가 설치되어 있지 않습니다: {exc}") from exc

    if not path.exists():
        raise PdfParseError(f"PDF 파일이 없습니다: {path}")

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise PdfParseError(f"PDF 를 열 수 없습니다: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise PdfParseError("PDF 에 페이지가 없습니다.")

        texts: list[str] = []
        images: list[bytes] = []
        pages_to_read = min(doc.page_count, max_pages)
        zoom = dpi / 72.0

        for page_no in range(pages_to_read):
            page = doc[page_no]
            texts.append(f"--- page {page_no + 1} ---\n{page.get_text()}")
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            images.append(pixmap.tobytes("png"))

        return ParsedPdf(
            text="\n\n".join(texts),
            page_images=images,
            page_count=doc.page_count,
            truncated=doc.page_count > max_pages,
        )
    finally:
        doc.close()
