"""
backends/wan.py
---------------
Alibaba Wan 백엔드 (Model Studio / DashScope, 비동기 작업 REST API).

흐름: POST .../video-generation/video-synthesis (X-DashScope-Async: enable)
  → output.task_id → GET .../tasks/{task_id} 폴링
  → output.task_status == SUCCEEDED → output.video_url 다운로드.

하나의 클래스로 text-to-video 와 image-to-video 를 모두 처리한다.
  - text-to-video : input.prompt + parameters.size("1920*1080")
  - image-to-video: input.img_url(첫 프레임 base64 data URI)
                    + parameters.resolution("1080P") — 종횡비는 입력 이미지 기준.
모델 ID 가 t2v/i2v 로 다르므로 등록 시 t2v_model / i2v_model 을 주입한다.

[주의]
- duration 은 모델마다 지원 값이 다르다(wan2.2: 5초 고정, wan2.5: 5/10).
  등록 시 supported_durations 로 주입하고 normalize_duration 으로 보정한다.
- 해상도(티어)도 모델마다 다르다(wan2.2: 480P/1080P, wan2.5: 480/720/1080).
  등록 시 resolutions 로 허용 티어를 주입하고 가장 가까운 값으로 보정한다.
- 오디오: wan2.5/2.6 계열은 오디오를 기본 생성(프롬프트의 보이스오버 발화),
  wan2.1/2.2 계열은 무음이다. has_audio 로 표기만 한다(API 플래그 없음).
- video_url 은 24시간만 유효하므로 즉시 다운로드해 out_path 에 저장한다.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path

import httpx

from ..config import Settings
from .base import ClipGenerationError, ClipResult, ClipSpec, VideoBackend

# 해상도 티어 순서(낮음→높음). 요청 티어가 미지원이면 가장 가까운 값으로 보정.
_TIER_ORDER = ("480p", "720p", "1080p")

# text-to-video 용 (종횡비, 티어) → "width*height" 문자열.
# DashScope 는 비율/티어가 아닌 명시적 픽셀 크기를 요구한다.
_T2V_SIZE = {
    ("16:9", "480p"): "832*480",
    ("9:16", "480p"): "480*832",
    ("16:9", "720p"): "1280*720",
    ("9:16", "720p"): "720*1280",
    ("16:9", "1080p"): "1920*1080",
    ("9:16", "1080p"): "1080*1920",
}


class WanBackend(VideoBackend):
    provider = "Alibaba (Model Studio)"
    description = "Alibaba Wan (DashScope). text/image-to-video, 비동기 작업."
    supported_durations = (5.0,)
    supports_image_input = True   # img_url (첫 프레임 이미지)

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.dashscope_api_key)

    # ------------------------------------------------------------------ #
    def normalize_duration(self, requested: float) -> float:
        durations = self.params.get("supported_durations") or self.supported_durations
        return min(durations, key=lambda d: abs(d - requested))

    def _allowed_tiers(self) -> tuple[str, ...]:
        return tuple(
            t.lower() for t in (self.params.get("resolutions") or _TIER_ORDER)
        )

    def _pick_tier(self, requested: str) -> str:
        """요청 해상도 티어를 모델이 허용하는 가장 가까운 값으로 보정."""
        req = (requested or "1080p").lower()
        if req not in _TIER_ORDER:
            req = "1080p"
        allowed = self._allowed_tiers()
        if req in allowed:
            return req
        ri = _TIER_ORDER.index(req)
        # 거리가 같으면 더 높은 티어를 선택(품질 우선). 720p+{480,1080} → 1080p.
        return min(
            allowed,
            key=lambda a: (abs(_TIER_ORDER.index(a) - ri), -_TIER_ORDER.index(a)),
        )

    # ------------------------------------------------------------------ #
    async def generate_clip(self, spec: ClipSpec, out_path: Path) -> ClipResult:
        duration = int(self.normalize_duration(spec.duration_sec))
        tier = self._pick_tier(spec.resolution)

        # 공통 input/parameters 구성. t2v/i2v 분기.
        input_obj: dict = {"prompt": spec.prompt}
        parameters: dict = {
            "duration": duration,
            "prompt_extend": self.params.get("prompt_extend", True),
            "watermark": False,   # 로고는 후처리 오버레이로 처리한다.
        }

        if spec.first_frame is not None:
            # image-to-video: 첫 프레임을 base64 data URI 로, resolution 티어로.
            model = self.params.get("i2v_model") or self.settings.wan_i2v_model_default
            input_obj["img_url"] = _to_data_uri(spec.first_frame)
            parameters["resolution"] = tier.upper()   # "1080p" → "1080P"
        else:
            # text-to-video: 명시적 size 픽셀 크기.
            model = self.params.get("t2v_model") or self.settings.wan_t2v_model_default
            aspect = spec.aspect_ratio if spec.aspect_ratio in ("16:9", "9:16") else "16:9"
            parameters["size"] = _T2V_SIZE[(aspect, tier)]

        payload = {"model": model, "input": input_obj, "parameters": parameters}
        base = self.settings.dashscope_base_url.rstrip("/")
        submit_url = f"{base}/services/aigc/video-generation/video-synthesis"
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1) 작업 제출 → task_id
            try:
                resp = await client.post(submit_url, json=payload, headers=headers)
                resp.raise_for_status()
                submitted = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(f"Wan 작업 제출 실패: {exc}") from exc

            out = submitted.get("output") or {}
            task_id = out.get("task_id")
            if not task_id:
                raise ClipGenerationError(
                    f"Wan 제출 응답에 task_id 가 없습니다: {submitted}"
                )

            # 2) 작업 폴링
            task_url = f"{base}/tasks/{task_id}"
            elapsed = 0.0
            video_url = None
            while True:
                if elapsed > self.settings.poll_timeout_sec:
                    raise ClipGenerationError(
                        f"Wan 폴링 시간 초과({self.settings.poll_timeout_sec:.0f}s)"
                    )
                await asyncio.sleep(self.settings.poll_interval_sec)
                elapsed += self.settings.poll_interval_sec
                try:
                    st_resp = await client.get(task_url, headers=headers)
                    st_resp.raise_for_status()
                    st_out = (st_resp.json().get("output") or {})
                except Exception as exc:  # noqa: BLE001
                    raise ClipGenerationError(f"Wan 상태 조회 실패: {exc}") from exc

                status = st_out.get("task_status")
                if status == "SUCCEEDED":
                    video_url = st_out.get("video_url")
                    break
                if status in ("PENDING", "RUNNING"):
                    continue
                # FAILED / CANCELED / UNKNOWN
                msg = st_out.get("message") or st_out.get("code") or status
                raise ClipGenerationError(f"Wan 생성 실패(status={status}): {msg}")

            if not video_url:
                raise ClipGenerationError("Wan 응답에 video_url 이 없습니다.")

            # 3) 비디오 다운로드 (URL 은 24시간만 유효)
            try:
                async with client.stream(
                    "GET", video_url, timeout=300.0
                ) as download:
                    download.raise_for_status()
                    with open(out_path, "wb") as fh:
                        async for chunk in download.aiter_bytes():
                            fh.write(chunk)
            except Exception as exc:  # noqa: BLE001
                raise ClipGenerationError(f"Wan 비디오 다운로드 실패: {exc}") from exc

        return ClipResult(
            path=out_path,
            duration_sec=float(duration),
            meta={"model": model, "backend_job_id": task_id, "video_url": video_url},
        )


# ---------------------------------------------------------------------- #
# image-to-video 헬퍼
# ---------------------------------------------------------------------- #
def _to_data_uri(image_path: Path) -> str:
    """첫 프레임 이미지를 base64 data URI 로 변환한다."""
    if not image_path.exists():
        raise ClipGenerationError(f"첫 프레임 이미지가 없습니다: {image_path}")
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    try:
        raw = image_path.read_bytes()
    except OSError as exc:
        raise ClipGenerationError(f"첫 프레임 이미지 읽기 실패: {exc}") from exc
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
