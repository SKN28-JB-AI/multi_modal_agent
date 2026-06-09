#!/usr/bin/env bash
# multi_modal_agent 컨테이너 실행(백그라운드).
#
# data/ 디렉토리를 호스트 사용자 소유로 둔 채 바인드 마운트할 때,
# 컨테이너를 'data/ 를 소유한 uid:gid' 로 실행해 권한 문제(Permission denied)를
# 원천 차단한다(이미지의 비루트 appuser uid 10001 과 달라도 OK).
#
# 사용: multi_modal_agent 디렉토리에서  ./run-docker.sh
set -euo pipefail

IMAGE="${IMAGE:-multi-modal-agent}"
NAME="${NAME:-multi-modal-agent}"
PORT="${PORT:-8000}"
DATA_DIR="$(pwd)/data"

# .env 필수(없으면 APP_KEYS 미설정으로 서버가 기동을 거부함)
if [ ! -f "$(pwd)/.env" ]; then
  echo "ERROR: .env 가 없습니다. 'cp .env.example .env' 후 값을 채우세요." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
# data/ 를 소유한 uid:gid 를 그대로 컨테이너 실행 사용자로 사용
USER_SPEC="$(stat -c '%u:%g' "$DATA_DIR")"

# 같은 이름의 기존 컨테이너 정리
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --user "$USER_SPEC" \
  -p "${PORT}:8000" \
  -v "$(pwd)/.env:/app/.env:ro" \
  -v "$DATA_DIR:/app/data" \
  --restart unless-stopped \
  "$IMAGE"

echo "started (user=$USER_SPEC) -> http://localhost:${PORT}/health"
echo "logs : docker logs -f $NAME"
echo "stop : docker stop $NAME"
