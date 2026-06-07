"""remix 기능 테스트 (POST /v1/jobs/{job_id}/remix)."""

from .conftest import auth_headers, make_sample_pdf, wait_for_job


def _create_message_job(client, model="mock") -> dict:
    resp = client.post(
        "/v1/videos/message",
        json={"prompt": "a calm beach", "model": model},
        headers=auth_headers(),
    )
    assert resp.status_code == 202
    return wait_for_job(client, resp.json()["job_id"])


def test_remix_message_job_end_to_end(client):
    source = _create_message_job(client)
    resp = client.post(
        f"/v1/jobs/{source['job_id']}/remix",
        json={"prompt": "make it a night city scene", "scene_index": 0},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    body = wait_for_job(client, resp.json()["job_id"])

    assert body["status"] == "completed", body
    assert body["mode"] == "remix"
    # 수정 프롬프트가 스토리보드에 기록됨
    assert body["storyboard"]["scenes"][0]["prompt"] == "make it a night city scene"

    video = client.get(body["video_url"], headers=auth_headers())
    assert video.status_code == 200
    assert len(video.content) > 1000


def test_remix_pdf_job_scene(client, tmp_path):
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock"},
        headers=auth_headers(),
    )
    source = wait_for_job(client, resp.json()["job_id"])
    assert source["status"] == "completed"

    # 씬 1만 remix
    resp = client.post(
        f"/v1/jobs/{source['job_id']}/remix",
        json={"prompt": "closer product shot", "scene_index": 1},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    body = wait_for_job(client, resp.json()["job_id"])

    assert body["status"] == "completed", body
    # 씬 0 프롬프트는 그대로, 씬 1만 교체됨
    assert body["storyboard"]["scenes"][0]["prompt"] == "sunny beach, product splash"
    assert body["storyboard"]["scenes"][1]["prompt"] == "closer product shot"
    # SRT 도 재생성됨 (원본 스토리보드의 카피 유지)
    assert body["subtitles_url"]
    srt = client.get(body["subtitles_url"], headers=auth_headers())
    assert "시원한 한 모금" in srt.text


def test_remix_requires_completed_job(client):
    # 실패한 잡은 remix 불가
    resp = client.post(
        "/v1/videos/message",
        json={"prompt": "x", "model": "mock-fail"},
        headers=auth_headers(),
    )
    failed = wait_for_job(client, resp.json()["job_id"])
    assert failed["status"] == "failed"

    resp = client.post(
        f"/v1/jobs/{failed['job_id']}/remix",
        json={"prompt": "y"},
        headers=auth_headers(),
    )
    assert resp.status_code == 409


def test_remix_unsupported_backend_rejected(client):
    source = _create_message_job(client, model="mock-noremix")
    assert source["status"] == "completed"
    resp = client.post(
        f"/v1/jobs/{source['job_id']}/remix",
        json={"prompt": "y"},
        headers=auth_headers(),
    )
    assert resp.status_code == 422
    assert "지원하지 않" in resp.json()["detail"]


def test_remix_bad_scene_index(client):
    source = _create_message_job(client)
    resp = client.post(
        f"/v1/jobs/{source['job_id']}/remix",
        json={"prompt": "y", "scene_index": 5},
        headers=auth_headers(),
    )
    assert resp.status_code == 422


def test_models_expose_supports_remix(client):
    models = {
        m["name"]: m
        for m in client.get("/v1/models", headers=auth_headers()).json()["models"]
    }
    assert models["sora-2"]["supports_remix"] is True
    assert models["sora-2-pro"]["supports_remix"] is True
    assert models["veo-3.1"]["supports_remix"] is False
    assert models["ltx-2.3"]["supports_remix"] is False
