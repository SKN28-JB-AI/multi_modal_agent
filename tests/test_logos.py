"""서버 logos/ 폴더 기반 로고 적용 테스트."""

from pathlib import Path

from .conftest import auth_headers, make_logo, make_sample_pdf, wait_for_job


def _post_pdf(client, tmp_path, options: str = None):
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    data = {"model": "mock"}
    if options:
        data["options"] = options
    return client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data=data,
        headers=auth_headers(),
    )


def test_list_logos(make_client, tmp_path):
    client = make_client()
    logos_dir = tmp_path / "logos"
    make_logo(logos_dir / "brand_a.png")
    make_logo(logos_dir / "default.png")

    resp = client.get("/v1/logos", headers=auth_headers())
    body = resp.json()
    assert set(body["logos"]) == {"brand_a.png", "default.png"}
    assert body["default"] == "default.png"


def test_list_logos_empty(make_client):
    client = make_client()
    resp = client.get("/v1/logos", headers=auth_headers())
    assert resp.json() == {"logos": [], "default": None}


def test_pdf_job_applies_named_logo(make_client, tmp_path):
    client = make_client()
    make_logo(tmp_path / "logos" / "brand_a.png")
    make_logo(tmp_path / "logos" / "brand_b.png")

    resp = _post_pdf(client, tmp_path, '{"logo_name": "brand_b.png"}')
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body

    # 선택된 로고가 잡 디렉터리에 복사되고 오버레이 산출물이 생성됨
    job_dir = tmp_path / "data" / "jobs" / job_id
    assert (job_dir / "logo.png").exists()
    assert (job_dir / "merged_branded.mp4").exists()


def test_pdf_job_uses_default_logo_when_unspecified(make_client, tmp_path):
    client = make_client()
    make_logo(tmp_path / "logos" / "default.png")
    make_logo(tmp_path / "logos" / "zz_other.png")

    resp = _post_pdf(client, tmp_path)
    job_id = resp.json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body
    job_dir = tmp_path / "data" / "jobs" / job_id
    assert (job_dir / "logo.png").exists()
    assert (job_dir / "merged_branded.mp4").exists()


def test_pdf_job_without_any_logo_still_completes(make_client, tmp_path):
    client = make_client()  # logos/ 비어 있음
    resp = _post_pdf(client, tmp_path)
    job_id = resp.json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body
    job_dir = tmp_path / "data" / "jobs" / job_id
    assert not (job_dir / "logo.png").exists()
    assert not (job_dir / "merged_branded.mp4").exists()


def test_unknown_logo_name_rejected_before_job_creation(make_client, tmp_path):
    client = make_client()
    make_logo(tmp_path / "logos" / "brand_a.png")
    resp = _post_pdf(client, tmp_path, '{"logo_name": "nope.png"}')
    assert resp.status_code == 422
    assert "brand_a.png" in resp.json()["detail"]


def test_logo_name_path_traversal_rejected(make_client, tmp_path):
    client = make_client()
    resp = _post_pdf(client, tmp_path, '{"logo_name": "../secret.png"}')
    assert resp.status_code == 422


def test_uploaded_logo_overrides_server_logo(make_client, tmp_path):
    client = make_client()
    make_logo(tmp_path / "logos" / "default.png")
    uploaded = tmp_path / "uploaded.png"
    make_logo(uploaded)

    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={
            "file": ("plan.pdf", pdf.read_bytes(), "application/pdf"),
            "logo": ("uploaded.png", uploaded.read_bytes(), "image/png"),
        },
        data={"model": "mock", "options": '{"logo_name": "default.png"}'},
        headers=auth_headers(),
    )
    job_id = resp.json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body
    job_dir = tmp_path / "data" / "jobs" / job_id
    # 업로드 로고가 사용됨 (서버 default.png 가 아니라 업로드 바이트와 일치)
    assert (job_dir / "logo.png").read_bytes() == uploaded.read_bytes()
