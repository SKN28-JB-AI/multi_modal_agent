"""generation_mode="single" — 단일 생성 요청 모드 테스트."""

from .conftest import MockBackend, auth_headers, make_sample_pdf, wait_for_job


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


def test_single_mode_is_default_and_makes_one_request(client, tmp_path):
    resp = _post_pdf(client, tmp_path)   # options 미지정 → single
    assert resp.status_code == 202, resp.text
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body

    # 생성 요청이 정확히 1번
    prompts = MockBackend.captured_prompts
    assert len(prompts) == 1

    # 샷 타임라인 프롬프트 구조 검증
    p = prompts[0]
    assert "Shot 1 (0-4s):" in p
    assert "Shot 2 (4-8s):" in p
    assert "sunny beach, product splash" in p
    assert "seamless transitions" in p
    # 보이스오버 3요소: 따옴표 대사 + 언어 + 화자
    assert '"시원한 하루를 시작하세요."' in p
    assert '"지금 만나보세요."' in p
    assert "speaks in Korean" in p

    # 클립(SceneState)은 1개, 스토리보드는 원본 씬 구조 유지
    assert len(body["scenes"]) == 1
    assert len(body["storyboard"]["scenes"]) == 2

    video = client.get(body["video_url"], headers=auth_headers())
    assert video.status_code == 200 and len(video.content) > 1000


def test_single_mode_srt_scaled_to_actual_duration(client, tmp_path):
    """계획 8초 → mock 실제 2초 생성. SRT 가 1초 단위로 비례 축소되어야 함."""
    resp = _post_pdf(client, tmp_path)
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body
    srt = client.get(body["subtitles_url"], headers=auth_headers())
    assert "00:00:00,000 --> 00:00:01,000" in srt.text   # 씬1 (4s→1s)
    assert "00:00:01,000 --> 00:00:02,000" in srt.text   # 씬2


def test_single_mode_remix_whole_video(client, tmp_path):
    """단일 모드 잡은 scene_index=0 으로 전체 영상 remix 가 가능해야 함."""
    resp = _post_pdf(client, tmp_path)
    source = wait_for_job(client, resp.json()["job_id"])
    assert source["status"] == "completed"

    resp = client.post(
        f"/v1/jobs/{source['job_id']}/remix",
        json={"prompt": "make it rainy night mood", "scene_index": 0},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body
    video = client.get(body["video_url"], headers=auth_headers())
    assert video.status_code == 200


def test_scenes_mode_still_available(client, tmp_path):
    resp = _post_pdf(client, tmp_path, '{"generation_mode": "scenes"}')
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body
    assert len(body["scenes"]) == 2
    assert len(MockBackend.captured_prompts) == 2
