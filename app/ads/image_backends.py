"""
ads/image_backends.py
---------------------
컷 첫 장면 이미지 생성 백엔드 추상화 + 레지스트리.

비디오 백엔드(app/backends)와 같은 패턴:
  1) ImageBackend 를 상속해 generate_image() 를 구현
  2) register_image_model("이름", 클래스, **고정파라미터) 로 등록
  3) GET /v2/ads/image-models 와 요청의 model 필드에 자동 노출

[규약]
- generate_image 는 원본 이미지 바이트(PNG/JPEG)를 반환한다.
- 호출자가 cover_resize 로 목표 해상도에 정확히 맞춘다
  (Sora input_reference 는 비디오 해상도와 픽셀 단위 일치가 필요).
"""

from __future__ import annotations

import abc
import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import httpx

from ..config import Settings


class ImageBackendNotConfigured(Exception):
    """필요한 API 키가 없어 이미지 백엔드를 사용할 수 없음."""


class ImageGenerationError(Exception):
    """이미지 생성 실패."""


@dataclass
class ImageSpec:
    """이미지 1장 생성 사양."""

    prompt: str
    aspect_ratio: str = "16:9"     # "16:9" | "9:16"
    index: int = 0                 # 컷 번호(로그용)


class ImageBackend(abc.ABC):
    """이미지 생성 백엔드 인터페이스."""

    provider: str = ""
    description: str = ""
    # 프롬프트 빌더 분기용 패밀리 ('openai' | 'gemini' | 'fal')
    family: str = "openai"

    def __init__(self, settings: Settings, **params) -> None:
        self.settings = settings
        self.params = params

    @abc.abstractmethod
    async def generate_image(self, spec: ImageSpec) -> bytes:
        """spec 대로 이미지를 생성해 원본 바이트를 반환한다."""

    @classmethod
    @abc.abstractmethod
    def is_configured(cls, settings: Settings) -> bool:
        """필요한 API 키가 설정되어 있는가."""


# ====================================================================== #
# OpenAI (gpt-image-2 / gpt-image-1)
# ====================================================================== #
class OpenAIImageBackend(ImageBackend):
    provider = "OpenAI"
    description = "OpenAI Images API (gpt-image 계열)."
    family = "openai"

    # 모델별 종횡비 → 생성 해상도.
    # gpt-image-2 는 16의 배수인 임의 해상도를 지원(비율 1:3~3:1).
    # gpt-image-1 은 고정 enum(1024x1024 / 1536x1024 / 1024x1536)만 지원.
    _SIZE_MAP = {
        "gpt-image-2": {"16:9": "1792x1008", "9:16": "1008x1792"},
        "gpt-image-1": {"16:9": "1536x1024", "9:16": "1024x1536"},
    }

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.openai_api_key)

    async def generate_image(self, spec: ImageSpec) -> bytes:
        model = self.params.get("model", "gpt-image-2")
        size_map = self._SIZE_MAP.get(model, self._SIZE_MAP["gpt-image-1"])
        size = size_map.get(spec.aspect_ratio, "1024x1024")
        return await asyncio.to_thread(self._generate_sync, model, size, spec)

    def _generate_sync(self, model: str, size: str, spec: ImageSpec) -> bytes:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        try:
            response = client.images.generate(
                model=model, prompt=spec.prompt, n=1, size=size,
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageGenerationError(f"OpenAI 이미지 생성 실패: {exc}") from exc

        data = getattr(response, "data", None)
        if not data:
            raise ImageGenerationError("OpenAI 이미지 응답이 비어 있습니다.")
        b64 = getattr(data[0], "b64_json", None)
        if not b64:
            # url 응답 모델 대비 폴백
            url = getattr(data[0], "url", None)
            if url:
                return _download_bytes(url)
            raise ImageGenerationError(
                "OpenAI 이미지 응답에 b64_json/url 이 없습니다."
            )
        try:
            return base64.b64decode(b64)
        except Exception as exc:  # noqa: BLE001
            raise ImageGenerationError(f"이미지 base64 디코드 실패: {exc}") from exc


# ====================================================================== #
# Google (Imagen, google-genai SDK)
# ====================================================================== #
class GeminiImageBackend(ImageBackend):
    provider = "Google"
    description = "Google Imagen (Gemini API)."
    family = "gemini"

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.gemini_api_key)

    async def generate_image(self, spec: ImageSpec) -> bytes:
        model = self.params.get("model") or self.settings.imagen_model_default
        return await asyncio.to_thread(self._generate_sync, model, spec)

    def _generate_sync(self, model: str, spec: ImageSpec) -> bytes:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImageGenerationError(
                f"google-genai 패키지가 없습니다: {exc}. "
                "pip install google-genai 후 다시 시도하세요."
            ) from exc

        client = genai.Client(api_key=self.settings.gemini_api_key)
        try:
            response = client.models.generate_images(
                model=model,
                prompt=spec.prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=spec.aspect_ratio,
                    output_mime_type="image/png",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageGenerationError(f"Imagen 이미지 생성 실패: {exc}") from exc

        images = getattr(response, "generated_images", None) or []
        if not images:
            raise ImageGenerationError("Imagen 응답에 생성된 이미지가 없습니다.")
        image_bytes = getattr(getattr(images[0], "image", None), "image_bytes", None)
        if not image_bytes:
            raise ImageGenerationError("Imagen 응답에 이미지 바이트가 없습니다.")
        return image_bytes


# ====================================================================== #
# fal.ai (FLUX 계열, 큐 REST API — app/backends/ltx.py 와 동일 패턴)
# ====================================================================== #
class FalImageBackend(ImageBackend):
    provider = "fal.ai"
    description = "fal.ai 호스팅 이미지 모델 (FLUX 계열). 저비용."
    family = "fal"

    _GEN_SIZE = {"16:9": (1280, 720), "9:16": (720, 1280)}

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.fal_api_key)

    async def generate_image(self, spec: ImageSpec) -> bytes:
        endpoint = self.params.get("endpoint") or self.settings.fal_image_endpoint_default
        width, height = self._GEN_SIZE.get(spec.aspect_ratio, (1280, 720))
        payload = {
            "prompt": spec.prompt,
            "image_size": {"width": width, "height": height},
            "num_images": 1,
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
                raise ImageGenerationError(f"fal.ai 이미지 제출 실패: {exc}") from exc

            status_url = submitted.get("status_url")
            response_url = submitted.get("response_url")
            if not status_url or not response_url:
                raise ImageGenerationError(f"fal.ai 제출 응답 형식 오류: {submitted}")

            # 2) 폴링
            elapsed = 0.0
            interval = min(self.settings.poll_interval_sec, 3.0)
            while True:
                if elapsed > self.settings.image_poll_timeout_sec:
                    raise ImageGenerationError(
                        f"fal.ai 이미지 폴링 시간 초과"
                        f"({self.settings.image_poll_timeout_sec:.0f}s)"
                    )
                try:
                    status_resp = await client.get(status_url, headers=headers)
                    status_resp.raise_for_status()
                    status = status_resp.json().get("status")
                except Exception as exc:  # noqa: BLE001
                    raise ImageGenerationError(f"fal.ai 상태 조회 실패: {exc}") from exc

                if status == "COMPLETED":
                    break
                if status in ("IN_QUEUE", "IN_PROGRESS"):
                    await asyncio.sleep(interval)
                    elapsed += interval
                    continue
                raise ImageGenerationError(f"fal.ai 이미지 생성 실패(status={status})")

            # 3) 결과 → 이미지 다운로드
            try:
                result_resp = await client.get(response_url, headers=headers)
                result_resp.raise_for_status()
                result = result_resp.json()
                image_url = result["images"][0]["url"]
            except Exception as exc:  # noqa: BLE001
                raise ImageGenerationError(f"fal.ai 결과 조회 실패: {exc}") from exc

            try:
                img_resp = await client.get(image_url, timeout=120.0)
                img_resp.raise_for_status()
                return img_resp.content
            except Exception as exc:  # noqa: BLE001
                raise ImageGenerationError(f"fal.ai 이미지 다운로드 실패: {exc}") from exc


# ====================================================================== #
# Alibaba Qwen-Image (Model Studio / DashScope, 동기 multimodal-generation)
# ====================================================================== #
class QwenImageBackend(ImageBackend):
    provider = "Alibaba (Model Studio)"
    description = "Alibaba Qwen-Image (DashScope). 텍스트→이미지, 동기 호출."
    family = "qwen"   # prompts.build_image_prompt 의 서술형 빌더 사용

    # 모델 계열별 (종횡비 → size). qwen-image-2.0 계열은 고해상도,
    # max/plus 계열은 지원 enum 이 다르다(문서 기준).
    _SIZE_2_0 = {"16:9": "2688*1536", "9:16": "1536*2688"}
    _SIZE_MAX_PLUS = {"16:9": "1664*928", "9:16": "928*1664"}

    @classmethod
    def is_configured(cls, settings: Settings) -> bool:
        return bool(settings.dashscope_api_key)

    def _size_for(self, model: str, aspect_ratio: str) -> str:
        table = self._SIZE_2_0 if model.startswith("qwen-image-2.0") else self._SIZE_MAX_PLUS
        return table.get(aspect_ratio, table["16:9"])

    async def generate_image(self, spec: ImageSpec) -> bytes:
        model = self.params.get("model") or self.settings.qwen_image_model_default
        size = self._size_for(model, spec.aspect_ratio)
        base = self.settings.dashscope_base_url.rstrip("/")
        url = f"{base}/services/aigc/multimodal-generation/generation"
        payload = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": spec.prompt}]}
                ]
            },
            "parameters": {
                "size": size,
                "prompt_extend": True,
                "watermark": False,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.settings.image_poll_timeout_sec) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise ImageGenerationError(f"Qwen-Image 생성 실패: {exc}") from exc

            # 성공: output.choices[0].message.content[0].image (URL)
            image_url = _extract_qwen_image_url(body)
            if not image_url:
                code = body.get("code")
                msg = body.get("message")
                raise ImageGenerationError(
                    f"Qwen-Image 응답에 이미지가 없습니다"
                    + (f" (code={code}, message={msg})" if code or msg else f": {body}")
                )
            try:
                img_resp = await client.get(image_url, timeout=120.0)
                img_resp.raise_for_status()
                return img_resp.content
            except Exception as exc:  # noqa: BLE001
                raise ImageGenerationError(
                    f"Qwen-Image 이미지 다운로드 실패: {exc}"
                ) from exc


def _extract_qwen_image_url(body: dict) -> str | None:
    """DashScope multimodal-generation 응답에서 이미지 URL 을 꺼낸다."""
    try:
        choices = (body.get("output") or {}).get("choices") or []
        content = (choices[0].get("message") or {}).get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                return item["image"]
    except (IndexError, AttributeError, TypeError):
        return None
    return None


# ====================================================================== #
# 공통 유틸
# ====================================================================== #
def _download_bytes(url: str) -> bytes:
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def target_pixel_size(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """(aspect_ratio, resolution) → 캐노니컬 목표 픽셀 크기."""
    table = {
        ("16:9", "720p"): (1280, 720),
        ("16:9", "1080p"): (1920, 1080),
        ("9:16", "720p"): (720, 1280),
        ("9:16", "1080p"): (1080, 1920),
    }
    return table.get((aspect_ratio, resolution), (1920, 1080))


def cover_resize(raw: bytes, width: int, height: int, out_path: Path) -> None:
    """
    이미지를 (width x height)로 cover-crop 리사이즈해 PNG 로 저장한다.
    (비율 유지 확대/축소 후 중앙 크롭 → 왜곡 없이 정확한 픽셀 크기 보장)
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageGenerationError(
            f"Pillow 패키지가 없습니다: {exc}. pip install Pillow 후 다시 시도하세요."
        ) from exc

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            sw, sh = img.size
            if sw <= 0 or sh <= 0:
                raise ImageGenerationError("생성된 이미지 크기가 잘못되었습니다.")
            scale = max(width / sw, height / sh)
            nw = max(width, round(sw * scale))
            nh = max(height, round(sh * scale))
            img = img.resize((nw, nh), Image.LANCZOS)
            left = (nw - width) // 2
            top = (nh - height) // 2
            img = img.crop((left, top, left + width, top + height))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="PNG")
    except ImageGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageGenerationError(f"이미지 리사이즈/저장 실패: {exc}") from exc


# ====================================================================== #
# 레지스트리
# ====================================================================== #
@dataclass
class _Registration:
    backend_cls: Type[ImageBackend]
    params: dict


_REGISTRY: dict[str, _Registration] = {}


def register_image_model(name: str, backend_cls: Type[ImageBackend],
                         **params) -> None:
    _REGISTRY[name] = _Registration(backend_cls=backend_cls, params=params)


def unregister_image_model(name: str) -> None:
    _REGISTRY.pop(name, None)


def available_image_models() -> list[str]:
    return list(_REGISTRY.keys())


def get_image_backend(name: str, settings: Settings) -> ImageBackend:
    reg = _REGISTRY.get(name)
    if reg is None:
        raise KeyError(
            f"등록되지 않은 이미지 모델입니다: '{name}'. "
            f"사용 가능: {', '.join(sorted(_REGISTRY))}"
        )
    if not reg.backend_cls.is_configured(settings):
        raise ImageBackendNotConfigured(
            f"'{name}' 이미지 백엔드에 필요한 API 키가 설정되지 않았습니다. "
            f".env 를 확인하세요."
        )
    return reg.backend_cls(settings, **reg.params)


def image_model_info(settings: Settings, default_model: str) -> list[dict]:
    """GET /v2/ads/image-models 응답용 메타데이터."""
    infos = []
    for name, reg in sorted(_REGISTRY.items()):
        cls = reg.backend_cls
        infos.append(
            {
                "name": name,
                "provider": cls.provider,
                "description": cls.description,
                "configured": cls.is_configured(settings),
                "default": name == default_model,
            }
        )
    return infos


# ---------------------------------------------------------------------- #
# 기본 이미지 모델 등록 (벤더 3종)
# ---------------------------------------------------------------------- #
register_image_model("gpt-image-2", OpenAIImageBackend, model="gpt-image-2")
register_image_model("gpt-image-1", OpenAIImageBackend, model="gpt-image-1")
register_image_model("imagen-4.0", GeminiImageBackend)   # settings.imagen_model_default
register_image_model("flux-dev", FalImageBackend)        # settings.fal_image_endpoint_default
register_image_model("flux-schnell", FalImageBackend,
                     endpoint="fal-ai/flux/schnell")
# Alibaba Qwen-Image (settings.qwen_image_model_default = qwen-image-2.0-pro)
register_image_model("qwen-image", QwenImageBackend)
register_image_model("qwen-image-plus", QwenImageBackend, model="qwen-image-plus")
