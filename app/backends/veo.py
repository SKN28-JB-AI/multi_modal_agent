"""
backends/veo.py
---------------
Google Veo 3.1 백엔드 (Gemini API, google-genai SDK).

흐름: generate_videos(제출) → operations.get 폴링 → files.download.
Veo 3.1 은 영상과 동기화된 오디오를 네이티브로 생성한다.

[주의]
- 모델 ID 는 프리뷰 단계라 변경될 수 있다 → .env 의 VEO_MODEL_DEFAULT /
  VEO_FAST_MODEL 로 덮어쓸 수 있게 했다.
- duration 파라미터는 SDK 버전에 따라 미지원일 수 있어, 거부되면
  길이 지정 없이 1회 재시도한다(기본 8초 생성).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..config import Settings
from .base import ClipGenerationError, ClipResult, ClipSpec, VideoBackend


class VeoBackend(VideoBackend):
    provider = "Google"
    description = "Google Veo 3.1 (Gemini API). 네이티브 오디오 포함."
    supported_durations = (4.0, 6.0, 8.0)
    supports_image_input = True   # image= (첫 프레임 이미지)

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.gemini_api_key)

    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        model = self.params.get("model") or self.settings.veo_model_default
        duration = int(self.normalize_duration(spec.duration_sec))
        return await asyncio.to_thread(
            self._generate_sync, model, duration, spec, out_path
        )

    # ------------------------------------------------------------------ #
    def _generate_sync(
        self, model: str, duration: int, spec: ClipSpec, out_path: Path
    ) -> ClipResult:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ClipGenerationError(
                f"google-genai 패키지가 없습니다: {exc}. "
                "pip install google-genai 후 다시 시도하세요."
            ) from exc

        client = genai.Client(api_key=self.settings.gemini_api_key)

        config_kwargs: dict = {
            "aspect_ratio": spec.aspect_ratio,
            "resolution": spec.resolution,
        }

        # image-to-video: 첫 프레임 이미지를 함께 전달한다.
        submit_kwargs: dict = {}
        if spec.first_frame is not None:
            if not spec.first_frame.exists():
                raise ClipGenerationError(
                    f"첫 프레임 이미지가 없습니다: {spec.first_frame}"
                )
            try:
                submit_kwargs["image"] = types.Image.from_file(
                    location=str(spec.first_frame)
                )
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(
                    f"첫 프레임 이미지 로드 실패: {exc}"
                ) from exc

        # 1) 제출 (duration 지원 여부가 SDK/모델 버전에 따라 달라 2단계 시도)
        operation = None
        last_exc: Exception | None = None
        for attempt_kwargs in (
            {**config_kwargs, "duration_seconds": duration},
            config_kwargs,
        ):
            try:
                operation = client.models.generate_videos(
                    model=model,
                    prompt=spec.prompt,
                    config=types.GenerateVideosConfig(**attempt_kwargs),
                    **submit_kwargs,
                )
                break
            except Exception as exc:  # noqa: BLE001 - 파라미터 거부 시 폴백
                last_exc = exc
                continue
        if operation is None:
            raise ClipGenerationError(f"Veo 작업 제출 실패: {last_exc}")

        # 2) 폴링
        deadline = time.monotonic() + self.settings.poll_timeout_sec
        while not operation.done:
            if time.monotonic() > deadline:
                raise ClipGenerationError(
                    f"Veo 폴링 시간 초과({self.settings.poll_timeout_sec:.0f}s)"
                )
            time.sleep(self.settings.poll_interval_sec)
            try:
                operation = client.operations.get(operation)
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(f"Veo 상태 조회 실패: {exc}") from exc

        if getattr(operation, "error", None):
            raise ClipGenerationError(f"Veo 생성 실패: {operation.error}")

        response = getattr(operation, "response", None)
        videos = getattr(response, "generated_videos", None) or []
        if not videos:
            raise ClipGenerationError("Veo 응답에 생성된 비디오가 없습니다.")

        # 3) 다운로드
        try:
            video = videos[0]
            client.files.download(file=video.video)
            video.video.save(str(out_path))
        except Exception as exc:  # noqa: BLE001
            raise ClipGenerationError(f"Veo 다운로드 실패: {exc}") from exc

        return ClipResult(
            path=out_path,
            duration_sec=float(duration),
            meta={"model": model},
        )
