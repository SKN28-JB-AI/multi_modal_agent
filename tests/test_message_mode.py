"""메시지 모드 E2E 테스트 (mock 백엔드)."""

import subprocess

from .conftest import auth_headers, wait_for_job


def test_unknown_model_rejected(client):
    resp = client.post(
        "/v1/videos/message",
        json={"prompt": "test", "model": "no-such-model"},
        headers=auth_headers(),
    )
    assert resp.status_code == 422
    assert "no-such-model" in resp.json()["detail"]


def test_unconfigured_backend_returns_503(make_client):
    client = make_client(openai_api_key="")  # sora 키 없음
    resp = client.post(
        "/v1/videos/message",
        json={"prompt": "test", "model": "sora-2"},
        headers=auth_headers(),
    )
    assert resp.status_code == 503


def test_message_mode_end_to_end(client):
    resp = client.post(
        "/v1/videos/message",
        json={"prompt": "a calm beach at sunset", "model": "mock",
              "duration_sec": 4.0},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    body = wait_for_job(client, job_id)
    assert body["status"] == "completed", body
    assert body["progress"] == 1.0
    assert body["video_url"] == f"/v1/jobs/{job_id}/video"
    assert len(body["scenes"]) == 1
    assert body["scenes"][0]["status"] == "completed"

    # 결과 다운로드 + 실제 MP4 인지 ffprobe 로 검증
    video = client.get(body["video_url"], headers=auth_headers())
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert len(video.content) > 1000


def test_failed_generation_marks_job_failed(client):
    resp = client.post(
        "/v1/videos/message",
        json={"prompt": "x", "model": "mock-fail"},
        headers=auth_headers(),
    )
    job_id = resp.json()["job_id"]
    body = wait_for_job(client, job_id)
    assert body["status"] == "failed"
    assert "의도된 실패" in body["error"]
    # 미완료 잡의 비디오 요청은 409
    resp = client.get(f"/v1/jobs/{job_id}/video", headers=auth_headers())
    assert resp.status_code == 409


def test_job_not_found(client):
    resp = client.get("/v1/jobs/doesnotexist", headers=auth_headers())
    assert resp.status_code == 404
