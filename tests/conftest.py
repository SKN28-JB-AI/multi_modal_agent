"""
테스트 공용 픽스처.

- MockBackend : FFmpeg lavfi 로 2초짜리 진짜 MP4(영상+오디오)를 만드는
                가짜 비디오 백엔드. 외부 API 호출 없이 E2E 검증 가능.
- FakeLLM     : PDF 브리프/스토리보드를 고정값으로 반환하는 가짜 LLM.
- make_client : 설정을 덮어쓴 TestClient 팩토리.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import backends  # noqa: E402
from app.backends.base import ClipResult, ClipSpec, VideoBackend  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas import PdfJobOptions, Scene, Storyboard  # noqa: E402


# ---------------------------------------------------------------------- #
# Mock 비디오 백엔드
# ---------------------------------------------------------------------- #
def make_test_clip(out_path: Path, seconds: float = 2.0,
                   color: str = "red", with_audio: bool = True) -> None:
    """FFmpeg lavfi 로 테스트용 MP4 를 생성한다."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={seconds}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


class MockBackend(VideoBackend):
    provider = "test"
    description = "테스트용 mock 백엔드"
    supported_durations = (2.0,)

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return True

    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        await asyncio.to_thread(make_test_clip, out_path, 2.0)
        return ClipResult(path=out_path, duration_sec=2.0, meta={"mock": True})


class FailingBackend(MockBackend):
    description = "항상 실패하는 백엔드(재시도/실패 경로 테스트)"

    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        raise RuntimeError("의도된 실패")


# ---------------------------------------------------------------------- #
# Fake LLM
# ---------------------------------------------------------------------- #
class FakeLLM:
    async def analyze_pdf(self, text: str, page_images: list[bytes]) -> dict:
        assert "테스트 음료" in text  # 추출 텍스트가 실제로 전달되는지 검증
        return {"product": "테스트 음료", "key_message": "시원하다"}

    async def make_storyboard(
        self, brief: dict, options: PdfJobOptions
    ) -> Storyboard:
        return Storyboard(
            title="테스트 광고",
            narration_script="시원한 하루를 시작하세요.",
            scenes=[
                Scene(index=0, prompt="sunny beach, product splash",
                      duration_sec=4, on_screen_text="시원한 한 모금"),
                Scene(index=1, prompt="product close-up, soft light",
                      duration_sec=4, on_screen_text="지금 만나보세요"),
            ],
        )


# ---------------------------------------------------------------------- #
# 클라이언트 팩토리
# ---------------------------------------------------------------------- #
APP_KEY = "test-key"


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    clients: list[TestClient] = []

    def _make(**overrides) -> TestClient:
        backends.register("mock", MockBackend)
        backends.register("mock-fail", FailingBackend)

        from app.pipeline import orchestrator as orch
        monkeypatch.setattr(orch, "get_llm", lambda settings: FakeLLM())

        kwargs = dict(
            app_keys=APP_KEY,
            data_dir=str(tmp_path / "data"),
            openai_api_key="sk-fake-for-tests",
            clip_retries=0,
            _env_file=None,
        )
        kwargs.update(overrides)
        settings = Settings(**kwargs)
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield _make

    for c in clients:
        c.__exit__(None, None, None)
    backends.unregister("mock")
    backends.unregister("mock-fail")


@pytest.fixture
def client(make_client):
    return make_client()


def auth_headers() -> dict:
    return {"X-App-Key": APP_KEY}


def wait_for_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    """잡이 완료/실패 상태가 될 때까지 폴링한다."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError(f"잡 {job_id} 이 {timeout}초 내에 끝나지 않음")


def make_sample_pdf(path: Path) -> None:
    """PyMuPDF 로 2페이지짜리 가짜 광고 기획서 PDF 를 만든다."""
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 100), "Ad Planning Document", fontsize=20)
    # 한글은 CJK 내장 폰트("korea")로 넣어야 텍스트 추출이 가능하다.
    page1.insert_text(
        (72, 140), "Product: 테스트 음료 (Test Beverage)", fontname="korea"
    )
    page1.insert_text((72, 160), "Key message: 시원하다", fontname="korea")
    page2 = doc.new_page()
    page2.insert_text((72, 100), "Scene ideas: beach, close-up")
    doc.save(str(path))
    doc.close()
