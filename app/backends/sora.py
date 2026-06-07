"""
backends/sora.py
----------------
OpenAI Videos API (Sora 2) 백엔드.

흐름: POST /videos(제출) → 폴링 → /videos/{id}/content 다운로드.
Sora 2 는 영상과 동기화된 오디오를 함께 생성한다.

[주의]
- Sora 2 API 는 2026-09 종료가 예고되어 있다. 이 백엔드는 마이그레이션
  완충용이며, 신규 트래픽은 veo/ltx 계열을 권장한다.
- seconds 는 문자열로 전달해야 하며 지원 값이 제한적이다(기본 4/8/12초).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..config import Settings
from .base import ClipGenerationError, ClipResult, ClipSpec, VideoBackend

# (resolution, aspect_ratio) → Sora size 문자열
_SIZE_MAP = {
    ("720p", "16:9"): "1280x720",
    ("720p", "9:16"): "720x1280",
    ("1080p", "16:9"): "1792x1024",   # sora-2-pro 전용 고해상도
    ("1080p", "9:16"): "1024x1792",
}


class SoraBackend(VideoBackend):
    provider = "OpenAI"
    description = "OpenAI Videos API (Sora 2). 2026-09 API 종료 예정."
    supported_durations = (4.0, 8.0, 12.0)

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.openai_api_key)

    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        model = self.params.get("model") or self.settings.sora_model_default
        seconds = str(int(self.normalize_duration(spec.duration_sec)))

        size = _SIZE_MAP.get((spec.resolution, spec.aspect_ratio), "1280x720")
        # 1080p 계열 size 는 sora-2-pro 전용. 기본 모델이면 720p 로 강등.
        if model == "sora-2" and spec.resolution == "1080p":
            size = _SIZE_MAP[("720p", spec.aspect_ratio)]

        # OpenAI SDK 는 동기 클라이언트를 스레드에서 돌린다
        # (videos API 의 바이너리 다운로드까지 검증된 경로를 그대로 사용).
        return await asyncio.to_thread(
            self._generate_sync, model, seconds, size, spec, out_path
        )

    # ------------------------------------------------------------------ #
    def _generate_sync(
        self, model: str, seconds: str, size: str, spec: ClipSpec, out_path: Path
    ) -> ClipResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)

        # 1) 제출
        try:
            video = client.videos.create(
                model=model, prompt=spec.prompt, seconds=seconds, size=size
            )
        except Exception as exc:  # noqa: BLE001
            raise ClipGenerationError(f"Sora 작업 제출 실패: {exc}") from exc

        # 2) 폴링
        deadline = time.monotonic() + self.settings.poll_timeout_sec
        while True:
            if time.monotonic() > deadline:
                raise ClipGenerationError(
                    f"Sora 폴링 시간 초과({self.settings.poll_timeout_sec:.0f}s)"
                )
            try:
                video = client.videos.retrieve(video.id)
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(f"Sora 상태 조회 실패: {exc}") from exc

            status = getattr(video, "status", "unknown")
            if status == "completed":
                break
            if status in ("queued", "in_progress"):
                time.sleep(self.settings.poll_interval_sec)
                continue
            err = getattr(getattr(video, "error", None), "message", None)
            raise ClipGenerationError(err or f"Sora 생성 실패(status={status})")

        # 3) 다운로드
        try:
            content = client.videos.download_content(video.id, variant="video")
            content.write_to_file(str(out_path))
        except Exception as exc:  # noqa: BLE001
            raise ClipGenerationError(f"Sora 다운로드 실패: {exc}") from exc

        return ClipResult(
            path=out_path,
            duration_sec=float(seconds),
            meta={"backend_job_id": video.id, "model": model, "size": size},
        )
