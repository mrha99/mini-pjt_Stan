#!/usr/bin/env bash
# 인덱싱 → API → UI 순서로 실행
set -euo pipefail
python -m src.indexer
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
