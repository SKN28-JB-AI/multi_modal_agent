"""
backends/ltx.py
---------------
LTX-2 백엔드 (fal.ai 호스팅, 큐 REST API).

흐름: POST queue.fal.run/{endpoint} → status_url 폴링 → response_url 에서
결과 조회 → video.url 다운로드. LTX-2 도 영상+오디오를 함께 생성한다.

[주의]
- duration 은 enum 이다(Pro: 6/8/10, Fast: 6~20 짝수). 등록 시
  supported_durations 파라미터로 주입한다.
- LTX 해상도 enum 은 1080p/1440p/2160p 라 720p 요청은 1080p 로 승격된다.
- 폴링 URL 은 직접 조립하지 않고 제출 응답의 status_url/response_url 을
  그대로 쓴다(fal 권장 방식 — 엔드포인트 구조 변화에 안전).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ..config import Settings
from .base import ClipGenerationError, ClipResult, ClipSpec, VideoBackend


class LtxBackend(VideoBackend):
    provider = "Lightricks (fal.ai)"
    description = "LTX-2.3 (fal.ai 호스팅). 저비용, 오디오 포함."
    supported_durations = (6.0, 8.0, 10.0)

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.fal_api_key)

    def normalize_duration(self, requested: float) -> float:
        durations = self.params.get("supported_durations") or self.supported_durations
        return min(durations, key=lambda d: abs(d - requested))

    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        endpoint = self.params.get("endpoint") or self.settings.ltx_endpoint_default
        duration = int(self.normalize_duration(spec.duration_sec))
        resolution = "1080p" if spec.resolution == "720p" else spec.resolution

        payload = {
            "prompt": spec.prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": spec.aspect_ratio,
            "fps": 25,
            "generate_audio": spec.generate_audio,
        }
        headers = {"Authorization": f"Key {self.settings.fal_api_key}"}
        submit_url = f"{self.settings.fal_queue_base}/{endpoint}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1) 제출
            try:
                resp = await client.post(submit_url, json=payload, headers=headers)
                resp.raise_for_status()
                submitted = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(f"fal.ai 제출 실패: {exc}") from exc

            status_url = submitted.get("status_url")
            response_url = submitted.get("response_url")
            if not status_url or not response_url:
                raise ClipGenerationError(
                    f"fal.ai 제출 응답 형식 오류: {submitted}"
                )

            # 2) 폴링
            elapsed = 0.0
            while True:
                if elapsed > self.settings.poll_timeout_sec:
                    raise ClipGenerationError(
                        f"fal.ai 폴링 시간 초과"
                        f"({self.settings.poll_timeout_sec:.0f}s)"
                    )
                try:
                    status_resp = await client.get(status_url, headers=headers)
                    status_resp.raise_for_status()
                    status = status_resp.json().get("status")
                except Exception as exc:  # noqa: BLE001
                    raise ClipGenerationError(
                        f"fal.ai 상태 조회 실패: {exc}"
                    ) from exc

                if status == "COMPLETED":
                    break
                if status in ("IN_QUEUE", "IN_PROGRESS"):
                    await asyncio.sleep(self.settings.poll_interval_sec)
                    elapsed += self.settings.poll_interval_sec
                    continue
                raise ClipGenerationError(f"fal.ai 생성 실패(status={status})")

            # 3) 결과 → 비디오 URL 다운로드
            try:
                result_resp = await client.get(response_url, headers=headers)
                result_resp.raise_for_status()
                result = result_resp.json()
                video_url = result["video"]["url"]
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(f"fal.ai 결과 조회 실패: {exc}") from exc

            try:
                async with client.stream(
                    "GET", video_url, timeout=300.0
                ) as download:
                    download.raise_for_status()
                    with open(out_path, "wb") as fh:
                        async for chunk in download.aiter_bytes():
                            fh.write(chunk)
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(
                    f"fal.ai 비디오 다운로드 실패: {exc}"
                ) from exc

        return ClipResult(
            path=out_path,
            duration_sec=float(duration),
            meta={"endpoint": endpoint, "video_url": video_url},
        )
