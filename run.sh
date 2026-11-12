#!/usr/bin/env bash
# 가격 DB·인덱싱(없을 때만) → FastAPI(8000) → Streamlit(8501) 순서로 실행한다.
# Docker(CMD)와 로컬(가상환경 활성화 후 bash run.sh) 공통.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

# 1) 가격 이력 DB (가짜 데이터): 없을 때만 생성
if [ ! -f data/prices.db ]; then
  echo "[run.sh] data/prices.db 없음 → 가격 이력 생성"
  "$PYTHON" -m src.seed_db
else
  echo "[run.sh] data/prices.db 있음 → 생성 생략"
fi

# 2) 리뷰 색인 (Chroma): 컬렉션이 비어 있을 때만 생성
CHUNKS="$("$PYTHON" -c "from src import indexer; print(indexer.get_collection().count())" 2>/dev/null | tail -n 1)"
if [ "${CHUNKS:-0}" = "0" ]; then
  echo "[run.sh] 리뷰 색인 없음 → 색인 생성"
  "$PYTHON" -m src.indexer
else
  echo "[run.sh] 리뷰 색인 있음 (청크 ${CHUNKS}개) → 생성 생략"
fi

# 3) API → UI. 둘 중 하나가 종료되면 나머지도 내리고 종료한다 (컨테이너 재시작 정책에 맡김).
"$PYTHON" -m uvicorn src.api:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!
"$PYTHON" -m streamlit run src/app.py --server.port "$UI_PORT" --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false &
UI_PID=$!

trap 'kill "$API_PID" "$UI_PID" 2>/dev/null || true' INT TERM
wait -n "$API_PID" "$UI_PID"
STATUS=$?
kill "$API_PID" "$UI_PID" 2>/dev/null || true
exit "$STATUS"
