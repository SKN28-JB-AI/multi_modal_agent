"""
main.py
-------
FastAPI 애플리케이션 팩토리.

uvicorn 실행:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
(또는 프로젝트 루트에서 python run.py)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings
from .ads.manager import AdJobManager
from .jobs import JobManager
from .pipeline.orchestrator import Orchestrator
from .routers import ads as ads_router
from .routers import jobs as jobs_router
from .routers import logos as logos_router
from .routers import models as models_router
from .routers import videos as videos_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate_runtime()

    app = FastAPI(
        title="multi_modal_agent",
        description=(
            "멀티모달 동영상 생성 서비스 — 메시지/PDF 기획서 모드, "
            "모델 교체 가능(sora-2 / veo-3.1 / ltx-2.3 등)"
        ),
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 공유 상태
    app.state.settings = settings
    app.state.job_manager = JobManager(settings.jobs_dir)
    app.state.orchestrator = Orchestrator(settings, app.state.job_manager)
    # 광고 파이프라인(/v2/ads)은 기존 잡과 분리된 저장소를 쓴다.
    app.state.ad_job_manager = AdJobManager(settings.ad_jobs_dir)

    # 라우터
    app.include_router(videos_router.router)
    app.include_router(jobs_router.router)
    app.include_router(models_router.router)
    app.include_router(logos_router.router)
    app.include_router(ads_router.router)

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": __version__}

    return app
