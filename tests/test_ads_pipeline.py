"""
광고 파이프라인(/v2/ads) 테스트.

핵심 검증 항목:
  - 1단계 스토리보드 생성(202 + 폴링) 및 정규화된 JSON 산출
  - 2단계 컷 이미지: 목표 해상도 정확 일치(cover-crop)
  - 3단계 비디오: ★ images 완료 전 호출 시 412, 첫 프레임 전달 확인
  - 4단계 기획서 PDF: images/videos 와 무관하게 실행 가능
  - 409(중복/재실행), 422(모델 오류), 503(키 미설정), 인증
"""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .conftest import APP_KEY, MockBackend, auth_headers

from app.ads import image_backends as img_b
from app.ads.image_backends import ImageBackend, ImageSpec
from app.ads.schemas import AdStoryboard, Cut, Music
from app.backends import ClipResult, ClipSpec, register, unregister
from app.config import Settings
from app.main import create_app


# ---------------------------------------------------------------------- #
# Mock 구성요소
# ---------------------------------------------------------------------- #
def _fake_storyboard() -> AdStoryboard:
    return AdStoryboard(
        project="JB 테스트 캠페인",
        concept="청년을 위한 새로운 금융",
        target="20-30대 사회초년생",
        mood=["밝은", "역동적인"],
        total_duration_sec=4,
        aspect_ratio="16:9",
        format="유튜브 인스트림",
        music=Music(genre="일렉트로닉 팝", bpm=120, key_moment="후렴 드랍"),
        cuts=[
            Cut(cut=1, timecode="00:00-00:02", duration_sec=2,
                title="오프닝", scene="도시 출근길", visual="햇살 가득한 거리",
                camera="와이드 트래킹", on_screen_text="새로운 시작",
                voiceover="새로운 하루", sfx="도시 앰비언스", transition="컷"),
            Cut(cut=2, timecode="00:02-00:04", duration_sec=2,
                title="로고 리빌", scene="브랜드 로고", visual="블루 그라데이션",
                camera="줌 인", on_screen_text="JB금융그룹",
                voiceover="JB와 함께", sfx="시그니처 사운드", transition="페이드"),
        ],
        cta="지금 가입하세요",
        logo="JB금융그룹",
    )


def _png_bytes(width: int = 640, height: int = 360, color=(200, 30, 30)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class MockImageBackend(ImageBackend):
    provider = "test"
    description = "테스트용 mock 이미지 백엔드"
    family = "openai"
    captured_specs: list[ImageSpec] = []

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return True

    async def generate_image(self, spec: ImageSpec) -> bytes:
        MockImageBackend.captured_specs.append(spec)
        return _png_bytes()


class FailingCut2ImageBackend(MockImageBackend):
    """컷 2 만 실패시켜 부분 실패 경로를 검증한다."""

    async def generate_image(self, spec: ImageSpec) -> bytes:
        if spec.index == 2:
            raise RuntimeError("의도된 이미지 실패")
        return _png_bytes()


class MockI2VBackend(MockBackend):
    """image-to-video 를 지원하는 mock 비디오 백엔드."""

    description = "테스트용 mock image-to-video 백엔드"
    supports_image_input = True
    captured_first_frames: list = []

    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        MockI2VBackend.captured_first_frames.append(spec.first_frame)
        assert spec.first_frame is not None and spec.first_frame.exists(), \
            "first_frame 이 전달되어야 한다"
        return await super().generate_clip(spec, out_path)


# ---------------------------------------------------------------------- #
# 픽스처
# ---------------------------------------------------------------------- #
@pytest.fixture
def ads_client(tmp_path, monkeypatch):
    """광고 파이프라인용 TestClient(LLM/이미지/비디오 모두 mock)."""
    clients: list[TestClient] = []

    def _make(storyboard_behavior: str = "ok", **overrides) -> TestClient:
        MockImageBackend.captured_specs = []
        MockI2VBackend.captured_first_frames = []
        MockBackend.captured_prompts = []

        register("mock-i2v", MockI2VBackend)
        register("mock-no-image", MockBackend)   # supports_image_input=False
        img_b.register_image_model("mock-image", MockImageBackend)
        img_b.register_image_model("mock-image-fail2", FailingCut2ImageBackend)

        from app.routers import ads as ads_router

        async def _fake_generate(settings, prompt, options):
            if storyboard_behavior == "fail":
                raise RuntimeError("의도된 스토리보드 실패")
            if storyboard_behavior == "slow":
                await asyncio.sleep(0.8)
            return _fake_storyboard()

        monkeypatch.setattr(ads_router, "generate_storyboard", _fake_generate)

        kwargs = dict(
            app_keys=APP_KEY,
            data_dir=str(tmp_path / f"data-{len(clients)}"),
            openai_api_key="sk-fake-for-tests",
            ad_image_model_default="mock-image",
            ad_video_model_default="mock-i2v",
            clip_retries=0,
            image_retries=0,
            _env_file=None,
        )
        kwargs.update(overrides)
        client = TestClient(create_app(Settings(**kwargs)))
        client.__enter__()
        clients.append(client)
        return client

    yield _make

    for c in clients:
        c.__exit__(None, None, None)
    unregister("mock-i2v")
    unregister("mock-no-image")
    img_b.unregister_image_model("mock-image")
    img_b.unregister_image_model("mock-image-fail2")


def _wait_stage(client: TestClient, job_id: str, stage: str,
                timeout: float = 30.0) -> dict:
    """해당 단계가 completed/failed 가 될 때까지 폴링한다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v2/ads/{job_id}", headers=auth_headers())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["stages"][stage]["status"] in ("completed", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"잡 {job_id} 의 {stage} 단계가 {timeout}초 내에 끝나지 않음")


def _make_job_with_storyboard(client: TestClient) -> str:
    resp = client.post(
        "/v2/ads/storyboards",
        json={"prompt": "청년 적금 신규 캠페인 광고"},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    body = _wait_stage(client, job_id, "storyboard")
    assert body["stages"]["storyboard"]["status"] == "completed", body
    return job_id


# ====================================================================== #
# 인증
# ====================================================================== #
def test_auth_required(ads_client):
    client = ads_client()
    assert client.get("/v2/ads").status_code == 401
    assert client.get(
        "/v2/ads", headers={"X-App-Key": "wrong"}
    ).status_code == 403


# ====================================================================== #
# 1단계: 스토리보드
# ====================================================================== #
def test_storyboard_creation(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    resp = client.get(f"/v2/ads/{job_id}/storyboard", headers=auth_headers())
    assert resp.status_code == 200
    sb = resp.json()
    assert sb["project"] == "JB 테스트 캠페인"
    assert len(sb["cuts"]) == 2
    assert sb["cuts"][0]["timecode"] == "00:00-00:02"


def test_storyboard_failure_marks_stage_failed(ads_client):
    client = ads_client(storyboard_behavior="fail")
    resp = client.post(
        "/v2/ads/storyboards",
        json={"prompt": "실패 테스트"},
        headers=auth_headers(),
    )
    job_id = resp.json()["job_id"]
    body = _wait_stage(client, job_id, "storyboard")
    assert body["stages"]["storyboard"]["status"] == "failed"
    assert "의도된" in body["stages"]["storyboard"]["error"]

    # 스토리보드 실패 잡에는 images/pdf 모두 412
    for path in ("images", "proposal"):
        r = client.post(f"/v2/ads/{job_id}/{path}", headers=auth_headers())
        assert r.status_code == 412, f"{path}: {r.text}"


def test_empty_prompt_rejected(ads_client):
    client = ads_client()
    resp = client.post(
        "/v2/ads/storyboards", json={"prompt": "   "}, headers=auth_headers()
    )
    assert resp.status_code == 422


def test_unknown_job_404(ads_client):
    client = ads_client()
    assert client.get("/v2/ads/nope", headers=auth_headers()).status_code == 404
    assert client.post(
        "/v2/ads/nope/images", headers=auth_headers()
    ).status_code == 404


# ====================================================================== #
# 게이팅 (★ 요구사항 3: videos 는 images 완료 후에만)
# ====================================================================== #
def test_videos_blocked_until_images_done(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    # images 를 건너뛰고 videos 호출 → 412
    resp = client.post(f"/v2/ads/{job_id}/videos", headers=auth_headers())
    assert resp.status_code == 412
    assert "images" in resp.json()["detail"]


def test_images_blocked_while_storyboard_running(ads_client):
    client = ads_client(storyboard_behavior="slow")
    resp = client.post(
        "/v2/ads/storyboards", json={"prompt": "느린 테스트"},
        headers=auth_headers(),
    )
    job_id = resp.json()["job_id"]
    # 스토리보드가 아직 in_progress → images 는 412
    resp = client.post(f"/v2/ads/{job_id}/images", headers=auth_headers())
    assert resp.status_code == 412
    _wait_stage(client, job_id, "storyboard")


def test_completed_stage_needs_force(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    r = client.post(f"/v2/ads/{job_id}/images", headers=auth_headers())
    assert r.status_code == 202
    _wait_stage(client, job_id, "images")

    # force 없이 재실행 → 409
    r = client.post(f"/v2/ads/{job_id}/images", headers=auth_headers())
    assert r.status_code == 409
    # force=true → 202
    r = client.post(
        f"/v2/ads/{job_id}/images?force=true", headers=auth_headers()
    )
    assert r.status_code == 202
    body = _wait_stage(client, job_id, "images")
    assert body["stages"]["images"]["status"] == "completed"


# ====================================================================== #
# 2단계: 컷 이미지
# ====================================================================== #
def test_images_generated_at_exact_target_size(ads_client):
    from PIL import Image

    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    resp = client.post(
        f"/v2/ads/{job_id}/images",
        json={"model": "mock-image"},
        headers=auth_headers(),
    )
    assert resp.status_code == 202
    body = _wait_stage(client, job_id, "images")
    assert body["stages"]["images"]["status"] == "completed", body
    assert body["image_model"] == "mock-image"
    assert [a["status"] for a in body["images"]] == ["completed", "completed"]

    # 다운로드 후 목표 해상도(16:9/1080p → 1920x1080) 정확 일치 확인
    img_resp = client.get(
        f"/v2/ads/{job_id}/images/1", headers=auth_headers()
    )
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(img_resp.content)) as img:
        assert img.size == (1920, 1080)

    # 프롬프트에 도메인 제약(얼굴/텍스트 금지)이 들어갔는지 확인
    assert MockImageBackend.captured_specs, "이미지 백엔드가 호출되어야 한다"
    assert "No human faces" in MockImageBackend.captured_specs[0].prompt


def test_unknown_image_model_422(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)
    resp = client.post(
        f"/v2/ads/{job_id}/images",
        json={"model": "no-such-image-model"},
        headers=auth_headers(),
    )
    assert resp.status_code == 422


def test_unconfigured_image_model_503(ads_client):
    # 스토리보드용 OpenAI 키는 있고, Gemini 키만 없는 상태에서 imagen 요청
    client = ads_client(gemini_api_key="")
    job_id = _make_job_with_storyboard(client)
    resp = client.post(
        f"/v2/ads/{job_id}/images",
        json={"model": "imagen-4.0"},
        headers=auth_headers(),
    )
    assert resp.status_code == 503


def test_storyboard_requires_openai_key_503(ads_client):
    client = ads_client(openai_api_key="")
    resp = client.post(
        "/v2/ads/storyboards", json={"prompt": "키 없음"},
        headers=auth_headers(),
    )
    assert resp.status_code == 503


def test_partial_image_failure_blocks_videos(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    resp = client.post(
        f"/v2/ads/{job_id}/images",
        json={"model": "mock-image-fail2"},
        headers=auth_headers(),
    )
    assert resp.status_code == 202
    body = _wait_stage(client, job_id, "images")
    assert body["stages"]["images"]["status"] == "failed"
    statuses = {a["cut"]: a["status"] for a in body["images"]}
    assert statuses == {1: "completed", 2: "failed"}   # 부분 산출물 보존

    # images 가 실패했으므로 videos 는 여전히 412
    resp = client.post(f"/v2/ads/{job_id}/videos", headers=auth_headers())
    assert resp.status_code == 412


def test_image_models_listing(ads_client):
    client = ads_client()
    resp = client.get("/v2/ads/image-models", headers=auth_headers())
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()["models"]}
    # 벤더 3종 기본 모델이 모두 노출되는지
    assert {"gpt-image-2", "gpt-image-1", "imagen-4.0", "flux-dev"} <= names
    defaults = [m for m in resp.json()["models"] if m["default"]]
    assert len(defaults) == 1 and defaults[0]["name"] == "mock-image"


# ====================================================================== #
# 3단계: 이미지 기반 비디오
# ====================================================================== #
def test_full_pipeline_videos_after_images(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    client.post(f"/v2/ads/{job_id}/images", headers=auth_headers())
    _wait_stage(client, job_id, "images")

    resp = client.post(
        f"/v2/ads/{job_id}/videos",
        json={"model": "mock-i2v"},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    body = _wait_stage(client, job_id, "videos", timeout=60.0)
    assert body["stages"]["videos"]["status"] == "completed", body
    assert body["video_model"] == "mock-i2v"
    assert [a["status"] for a in body["videos"]] == ["completed", "completed"]

    # 첫 프레임 이미지가 컷마다 전달됐는지
    assert len(MockI2VBackend.captured_first_frames) == 2
    assert all(p is not None for p in MockI2VBackend.captured_first_frames)

    # 컷 클립 + 최종 결합본 다운로드
    clip = client.get(f"/v2/ads/{job_id}/videos/1", headers=auth_headers())
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/mp4"

    assert body["final_video_url"] == f"/v2/ads/{job_id}/video"
    final = client.get(body["final_video_url"], headers=auth_headers())
    assert final.status_code == 200
    assert len(final.content) > 1000


def test_video_model_without_image_input_rejected(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)
    client.post(f"/v2/ads/{job_id}/images", headers=auth_headers())
    _wait_stage(client, job_id, "images")

    resp = client.post(
        f"/v2/ads/{job_id}/videos",
        json={"model": "mock-no-image"},
        headers=auth_headers(),
    )
    assert resp.status_code == 422
    assert "image-to-video" in resp.json()["detail"]


# ====================================================================== #
# 4단계: 기획서 PDF (2·3단계와 무관 — 요구사항 4)
# ====================================================================== #
def test_proposal_independent_of_images_and_videos(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    # images/videos 없이 곧바로 proposal → 성공해야 한다
    resp = client.post(f"/v2/ads/{job_id}/proposal", headers=auth_headers())
    assert resp.status_code == 202, resp.text
    body = _wait_stage(client, job_id, "pdf")
    assert body["stages"]["pdf"]["status"] == "completed", body
    assert body["pdf_url"] == f"/v2/ads/{job_id}/proposal"

    pdf = client.get(body["pdf_url"], headers=auth_headers())
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content[:5] == b"%PDF-"
    assert len(pdf.content) > 1000


def test_proposal_includes_cut_images_when_available(ads_client):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)

    # 텍스트만 있는 PDF
    client.post(f"/v2/ads/{job_id}/proposal", headers=auth_headers())
    _wait_stage(client, job_id, "pdf")
    text_only = client.get(
        f"/v2/ads/{job_id}/proposal", headers=auth_headers()
    ).content

    # 이미지 생성 후 재생성 → 이미지가 삽입되어 파일이 커져야 한다
    client.post(f"/v2/ads/{job_id}/images", headers=auth_headers())
    _wait_stage(client, job_id, "images")
    client.post(
        f"/v2/ads/{job_id}/proposal?force=true", headers=auth_headers()
    )
    _wait_stage(client, job_id, "pdf")
    with_images = client.get(
        f"/v2/ads/{job_id}/proposal", headers=auth_headers()
    ).content

    assert len(with_images) > len(text_only)


# ====================================================================== #
# 영속화/복구
# ====================================================================== #
def test_job_persisted_to_disk(ads_client, tmp_path):
    client = ads_client()
    job_id = _make_job_with_storyboard(client)
    job_file = tmp_path / "data-0" / "ad_jobs" / job_id / "job.json"
    assert job_file.exists()
    assert "JB 테스트 캠페인" in job_file.read_text(encoding="utf-8")
