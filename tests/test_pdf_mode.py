"""PDF 기획서 모드 E2E 테스트 (mock 백엔드 + fake LLM)."""

from .conftest import auth_headers, make_sample_pdf, wait_for_job


def test_invalid_pdf_rejected(client):
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        data={"model": "mock"},
        headers=auth_headers(),
    )
    assert resp.status_code == 422


def test_pdf_mode_requires_llm_key(make_client, tmp_path):
    client = make_client(openai_api_key="")
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock"},
        headers=auth_headers(),
    )
    assert resp.status_code == 503


def test_bad_options_rejected(client, tmp_path):
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock", "options": '{"max_scenes": 99}'},
        headers=auth_headers(),
    )
    assert resp.status_code == 422


def test_pdf_mode_end_to_end(client, tmp_path):
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)

    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock",
              "options": '{"target_total_duration_sec": 8, "max_scenes": 2}'},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body

    # 스토리보드(② 단계 산출물)가 응답에 포함
    sb = body["storyboard"]
    assert sb["title"] == "테스트 광고"
    assert len(sb["scenes"]) == 2
    assert all(s["status"] == "completed" for s in body["scenes"])

    # 결합본 다운로드
    video = client.get(body["video_url"], headers=auth_headers())
    assert video.status_code == 200
    assert len(video.content) > 1000

    # 씬 카피(on_screen_text)가 있으므로 SRT 가 생성되어야 함
    assert body["subtitles_url"]
    srt = client.get(body["subtitles_url"], headers=auth_headers())
    assert srt.status_code == 200
    assert "시원한 한 모금" in srt.text
    assert "00:00:02,000" in srt.text  # mock 클립은 2초 → 두 번째 자막 시작점


def test_pdf_parser_extracts_text_and_images(tmp_path):
    from app.pipeline.pdf_parser import extract_pdf

    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    parsed = extract_pdf(pdf, max_pages=10, dpi=72)
    assert "테스트 음료" in parsed.text
    assert parsed.page_count == 2
    assert len(parsed.page_images) == 2
    assert parsed.page_images[0].startswith(b"\x89PNG")
