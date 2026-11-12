# 진짜후기 쇼핑 비교 에이전트 — 클린 환경 재현용 단일 이미지 (PRD "Docker" 설계 메모)
#
#   docker build -t mini-pjt-shopping .
#   docker run --env-file .env -p 8000:8000 -p 8501:8501 -v "$(pwd)/data:/app/data" mini-pjt-shopping
#
# - .env(키·모델 ID)는 이미지에 넣지 않는다(.dockerignore). 실행 시 --env-file로 주입한다.
# - torch는 CPU 전용 빌드만 설치해 CUDA·GPU 패키지가 들어가지 않게 한다.
# - 의존성은 requirements.txt 고정 목록을 --no-deps로 설치한다(사유는 requirements.txt 상단 주석).
# - 임베딩·리랭커 모델은 빌드 단계에서 내려받아 이미지에 포함하고, 실행 시에는 오프라인으로 읽는다.
FROM python:3.14.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 1) torch: CPU 전용 인덱스에서만 설치 (GPU용 nvidia-*·triton 미포함)
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu

# 2) 나머지 고정 목록 (torch는 이미 설치되어 건너뜀)
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

# 3) Linux에서 빠진 의존성이 없는지 확인 (Windows에서 고정한 목록이므로).
#    scikit-network는 ragas가 요구하지만 Python 3.14 wheel이 없고 import하지 않아 제외했다.
RUN pip check > /tmp/pip-check.txt || true; \
    grep -viE "scikit-network|No broken requirements" /tmp/pip-check.txt > /tmp/pip-broken.txt || true; \
    if [ -s /tmp/pip-broken.txt ]; then echo "누락된 의존성:"; cat /tmp/pip-broken.txt; exit 1; fi; \
    if pip list --format=freeze | grep -iE "^(nvidia|triton)"; then echo "GPU 패키지가 설치됨"; exit 1; fi

# 4) 임베딩·리랭커 모델 미리 내려받기 (src/indexer.py EMBEDDING_MODEL, src/retriever.py RERANKER_MODEL)
#    sentence-transformers가 실제로 읽는 파일만 받는다 (ONNX·OpenVINO 등 중복 가중치 제외).
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-base
ARG RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${EMBEDDING_MODEL}', allow_patterns=['*.json', 'model.safetensors', 'sentencepiece.bpe.model', '1_Pooling/*']); \
snapshot_download('${RERANKER_MODEL}', allow_patterns=['*.json', 'model.safetensors', 'sentencepiece.bpe.model'])"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 5) 애플리케이션 (data/chroma·data/prices.db는 run.sh가 없을 때만 생성)
COPY . .
RUN sed -i 's/\r$//' run.sh && chmod +x run.sh && \
    python -c "import src.api, src.app; from src import indexer, retriever; \
indexer.E5Embedder().embed_query('빌드 확인'); retriever.CrossEncoderReranker().score('빌드', ['확인']); \
print('import·모델 오프라인 로딩 확인 OK')"

EXPOSE 8000 8501
# 시작 시 모델 예열(약 20초)이 끝나야 /health가 응답한다
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["bash", "run.sh"]
