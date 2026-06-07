"""
backends/sora.py
----------------
OpenAI Videos API (Sora 2) 백엔드.

흐름: POST /videos(제출) → 폴링 → /videos/{id}/content 다운로드.
Sora 2 는 영상과 동기화된 오디오를 함께 생성한다.

remix 지원: POST /videos/{video_id}/remix 로 기존 생성물을 프롬프트로
부분 수정한다(구도·연속성 유지). remix 결과도 일반 생성과 같은
폴링/다운로드 흐름을 따른다.

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
    supports_remix = True

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.openai_api_key)

    # ================================================================== #
    # 생성
    # ================================================================== #
    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        model = self.params.get("model") or self.settings.sora_model_default
        seconds = str(int(self.normalize_duration(spec.duration_sec)))

        size = _SIZE_MAP.get((spec.resolution, spec.aspect_ratio), "1280x720")
        # 1080p 계열 size 는 sora-2-pro 전용. 기본 모델이면 720p 로 강등.
        if model == "sora-2" and spec.resolution == "1080p":
            size = _SIZE_MAP[("720p", spec.aspect_ratio)]

        # OpenAI SDK 는 동기 클라이언트를 스레드에서 돌린다.
        return await asyncio.to_thread(
            self._generate_sync, model, seconds, size, spec, out_path
        )

    def _generate_sync(
        self, model: str, seconds: str, size: str, spec: ClipSpec, out_path: Path
    ) -> ClipResult:
        client = self._client()
        try:
            video = client.videos.create(
                model=model, prompt=spec.prompt, seconds=seconds, size=size
            )
        except Exception as exc:  # noqa: BLE001
            raise ClipGenerationError(f"Sora 작업 제출 실패: {exc}") from exc

        video = self._wait_until_complete(client, video)
        self._download(client, video.id, out_path)
        return ClipResult(
            path=out_path,
            duration_sec=float(seconds),
            meta={"backend_job_id": video.id, "model": model, "size": size},
        )

    # ================================================================== #
    # remix : 기존 생성물 부분 수정
    # ================================================================== #
    async def remix_clip(
        self, source_video_id: str, prompt: str, out_path: Path
    ) -> ClipResult:
        return await asyncio.to_thread(
            self._remix_sync, source_video_id, prompt, out_path
        )

    def _remix_sync(
        self, source_video_id: str, prompt: str, out_path: Path
    ) -> ClipResult:
        client = self._client()
        try:
            video = client.videos.remix(video_id=source_video_id, prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            raise ClipGenerationError(
                f"Sora remix 제출 실패(video_id={source_video_id}): {exc}"
            ) from exc

        video = self._wait_until_complete(client, video)
        self._download(client, video.id, out_path)
        try:
            seconds = float(getattr(video, "seconds", 0) or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        return ClipResult(
            path=out_path,
            duration_sec=seconds,
            meta={
                "backend_job_id": video.id,
                "remixed_from": source_video_id,
            },
        )

    # ================================================================== #
    # 공통 헬퍼
    # ================================================================== #
    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.settings.openai_api_key)

    def _wait_until_complete(self, client, video):
        """폴링으로 완료를 기다린다. 실패/타임아웃 시 ClipGenerationError."""
        deadline = time.monotonic() + self.settings.poll_timeout_sec
        while True:
            if time.monotonic() > deadline:
                raise ClipGenerationError(
                    f"Sora 폴링 시간 초과({self.settings.poll_timeout_sec:.0f}s)"
                )
            status = getattr(video, "status", "unknown")
            if status == "completed":
                return video
            if status in ("queued", "in_progress"):
                time.sleep(self.settings.poll_interval_sec)
                try:
                    video = client.videos.retrieve(video.id)
                except Exception as exc:  # noqa: BLE001
                    raise ClipGenerationError(
                        f"Sora 상태 조회 실패: {exc}"
                    ) from exc
                continue
            err = getattr(getattr(video, "error", None), "message", None)
            raise ClipGenerationError(err or f"Sora 생성 실패(status={status})")

    def _download(self, client, video_id: str, out_path: Path) -> None:
        try:
            content = client.videos.download_content(video_id, variant="video")
            content.write_to_file(str(out_path))
        except Exception as exc:  # noqa: BLE001
            raise ClipGenerationError(f"Sora 다운로드 실패: {exc}") from exc
