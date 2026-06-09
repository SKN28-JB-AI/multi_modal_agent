# syntax=docker/dockerfile:1

# ==================================================================== #
# multi_modal_agent — 멀티스테이지 Dockerfile
#
#   builder : 의존성 빌드/설치 전용 (컴파일러 포함, 이미지에 안 남김)
#   runtime : 실행 전용 슬림 이미지 (FFmpeg + 설치된 venv 만)
#
# 환경변수: 런타임에 마운트하는 '소스의 .env' 를 가장 우선해서 사용한다.
#   - .env 는 이미지에 굽지 않는다(.dockerignore 로 제외, 시크릿 보호).
#   - 실행 시 호스트의 .env 를 /app/.env 로 마운트한다.
#   - 우선순위(높음->낮음): init > .env(dotenv) > 컨테이너 env > 기본값.
#     (app/config.py 의 settings_customise_sources 로 보장)
#
# 실행 예:
#   docker build -t multi-modal-agent .
#   docker run --rm -p 8000:8000 -v "$(pwd)/.env:/app/.env:ro" \
#              -v "$(pwd)/data:/app/data" multi-modal-agent
# (docker compose 파일은 이후 별도 작성 예정)
# ==================================================================== #

ARG PYTHON_VERSION=3.11

# ----------------------------- builder ------------------------------ #
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# 일부 의존성의 sdist 컴파일을 대비한 빌드 도구 (런타임 이미지에는 미포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 격리된 virtualenv 에 의존성 설치 -> 런타임으로 통째 복사
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------------- runtime ------------------------------ #
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# 후처리 필수: FFmpeg(+ffprobe) — 반드시 설치되어야 함.
# CJK 자막/기획서 한글 폰트는 best-effort(없어도 빌드 진행; 런타임에
# SUBTITLE_FONT_PATH 로 폰트 지정 가능). 패키지명/릴리스 차이에 견고하게 대응.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && ( apt-get install -y --no-install-recommends fonts-nanum \
         || apt-get install -y --no-install-recommends fonts-noto-cjk \
         || echo "WARN: CJK 폰트 패키지 설치 실패 — 런타임 SUBTITLE_FONT_PATH 로 지정하세요" ) \
    && rm -rf /var/lib/apt/lists/*

# 빌더에서 만든 venv 만 복사(컴파일러/캐시 없음 -> 작고 안전)
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 비루트 사용자로 실행
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# 애플리케이션 소스만 복사 (.env 는 .dockerignore 로 제외 -> 런타임 마운트)
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser logos/ ./logos/
COPY --chown=appuser:appuser run.py ./run.py

# 잡/클립/최종본 산출물 디렉토리(운영 시 볼륨 마운트 권장)
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# /health 는 인증 불필요(공개 엔드포인트) -> 헬스체크에 사용
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

# 개발용 run.py 는 reload=True 라, 컨테이너에선 uvicorn 을 직접 실행한다.
# WORKDIR(/app)가 CWD 이므로 pydantic 이 /app/.env 를 읽는다.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
