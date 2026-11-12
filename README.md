# 미니 PJT: 진짜후기 쇼핑 비교 에이전트

## 무엇을 푸나
최저가와 리뷰는 넘쳐나지만 협찬 리뷰와 결제 조건 때문에 "나한테 진짜 맞는 상품"을 고르기 어려운 문제를, 협찬 리뷰를 걸러낸 실구매 후기와 내 조건 기준 실결제가로 서로 다른 모델 상위 3개를 추천하고 모델마다 판매처별 실결제가를 비교해 해결한다.

## 활용한 패턴 (Day 1~7)
<!-- TODO: 각 패턴을 배운 Day 번호로 X를 바꾸세요 -->
- Day 1: LCEL chain (Pydantic 구조화 출력) — 리뷰 분석·추천 카드를 `prompt | llm.with_structured_output(ReviewAnalysis)` 체인으로 생성하고, 같은 Pydantic 모델을 API 응답 스키마로 재사용
- Day 3: ReAct (도구 자율 선택) — 에이전트가 질문에 따라 상품 검색·리뷰 검색·가격 계산 도구를 스스로 골라 호출 (요청당 최대 6회)
- Day 2: RAG (하이브리드·리랭킹·쿼리 확장) — 리뷰를 BM25 + 벡터 앙상블로 검색(후보 모델로 범위 제한)하고 Cross-encoder로 리랭킹, 사용자 조건어로 멀티 쿼리 확장, 협찬 리뷰 청크는 검색 후보에서 제외
- Day 4: 도구 다중 (DB·계산기·외부 API) — 네이버 블로그 검색 API(NAVER API HUB), 상품 카탈로그(샘플 JSON), 가격 이력 SQLite, 실결제가 계산기
- Day 6: 가드레일 (PII·프롬프트 인젝션) — 입력과 리뷰 본문의 인젝션 문구 탐지·제거, 응답의 PII 마스킹, 구매 대행·규제 품목·범위 밖 요청 거절
- Day 11: Observability · Trace — LangSmith로 체인·도구·에이전트 호출 전 과정 기록
- Day 12: 평가 (RAGAS · LLM-as-Judge) — RAGAS 4지표와 추천 근거 타당성 Judge 채점으로 1·2차 자체 평가
- Docker · API — 표준 규약 `POST /query`(question → answer·contexts·trace)로 에이전트를 서비스하고, Dockerfile·run.sh로 클린 환경에서 재현

## 아키텍처

```
[Streamlit UI (src/app.py)]
          │  POST /query
          ▼
[FastAPI (src/api.py)]
          │
          ▼
[가드레일: 인젝션 차단 / PII 마스킹 / 범위 밖·규제 품목 거절 (src/guardrails.py)]
          │
          ▼
[ReAct 에이전트 (src/agent.py)] ──► [LCEL 체인 → Pydantic 추천 카드 (src/chains.py)]
          │ 도구 자율 선택
          ▼
┌─────────────── 도구 5종 (src/tools.py) ───────────────┐
│ 상품 검색 (샘플 JSON) │ 블로그 검색 API │ 리뷰 RAG     │
│ 가격 DB (SQLite) │ 실결제가 계산기                    │
└───────────────────────────────────────────────────────┘

전 구간 trace → LangSmith
전체 실행 환경 → Docker 단일 이미지 (run.sh)

[평가 파이프라인 (src/evaluate.py)]
  evaluation/test_queries.csv 30문항으로 POST /query 호출
  → RAGAS 4지표 + LLM-as-Judge 채점 → round1/round2 리포트
```

추천 요청 한 건의 흐름: ① 조건 파악(부족하면 되묻기) → ② 상품 수집 → ③ 리뷰 수집 → ④ 협찬 분리·조건별 요약 → ⑤ 점수화·실결제가 계산 → ⑥ 근거·출처가 달린 Top 3 추천(서로 다른 모델 3개, 카드마다 판매처별 실결제가 비교). 조건에 맞는 모델이 3개 미만이면 검색어를 바꿔 ②로 돌아간다(최대 2회).

## 실행 방법

### 0. 요구 사항
- Python 3.14.8 (로컬 실행 시)
- Docker (클린 환경 재현 시, 이미지 안에서 Python 3.14.8 사용)
- AWS 계정과 Amazon Bedrock의 Claude 모델 접근 권한 (LLM은 AWS Bedrock의 Claude 사용)

### 1. 환경 변수
프로젝트 루트에 `.env` 파일을 만들고 아래 값을 채운다. (키는 저장소에 커밋하지 않는다)

```
NAVER_CLIENT_ID=...            # NAVER API HUB 블로그 검색 키 ID (헤더 X-NCP-APIGW-API-KEY-ID)
NAVER_CLIENT_SECRET=...        # NAVER API HUB 블로그 검색 키 (헤더 X-NCP-APIGW-API-KEY)
REVIEW_SOURCE=sample           # blog: 블로그 API 실데이터("카테고리 + 조건어" 검색, 참고 후기로 별도 표시) / sample: 샘플 리뷰 (평가는 sample)
RETRIEVER_MODE=vector          # vector: 1차 기준선 / hybrid: BM25+벡터+리랭킹+협찬 제외 (2차)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...                 # Claude 모델을 사용할 수 있는 Bedrock 리전
BEDROCK_MODEL_ID=...           # 사용할 Claude 모델 ID (답변 생성)
JUDGE_MODEL_ID=...             # 평가용 Judge 모델 ID (생성 모델과 다른 모델, RAGAS·LLM-as-Judge)
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=mini-pjt-shopping-agent

# 선택 (기본값 있음)
LLM_MAX_CONCURRENCY=3          # Bedrock 동시 호출 수 상한
LLM_MAX_ATTEMPTS=5             # LLM 일시 오류(분 단위 호출 한도 등) 재시도 횟수. 일일 토큰 한도는 재시도하지 않음
LLM_RETRY_MAX_DELAY=8          # 재시도 대기 상한(초)
API_WARMUP=1                   # 0이면 API 시작 시 임베딩·리랭커 예열 생략
```

`.env`는 `docker run --env-file`로 그대로 읽히므로 값 뒤에 주석이나 따옴표를 붙이지 않는다(위 예시의 `#` 주석은 설명용).

### 2-A. Docker로 실행 (클린 환경 재현)
```bash
docker build -t mini-pjt-shopping .
docker run --env-file .env -p 8000:8000 -p 8501:8501 \
  -v "$(pwd)/data:/app/data" mini-pjt-shopping   # Chroma 인덱스·SQLite를 호스트에 보존 (이미지 WORKDIR은 /app)
```

Windows PowerShell에서는 볼륨 경로를 `${PWD}`로 쓴다.
```powershell
docker build -t mini-pjt-shopping .
docker run --env-file .env -p 8000:8000 -p 8501:8501 `
  -v "${PWD}/data:/app/data" mini-pjt-shopping
```

- 이미지: `python:3.14.8-slim` + CPU 전용 torch(`download.pytorch.org/whl/cpu`, GPU 패키지 없음) + 고정 목록 `--no-deps` 설치 + 임베딩·리랭커 모델(빌드 시 내려받아 포함, 실행 시 오프라인 로딩).
- 측정(2026-11-12, Docker 29.9.3): 첫 빌드 약 31분(모델 다운로드 약 23분, 패키지 설치 약 7분), 이미지 5.78GB(모델 3.42GB, 패키지 1.41GB). 코드만 바꾸면 마지막 두 단계만 다시 빌드된다.
- `.env`는 이미지에 넣지 않고(`.dockerignore`) `--env-file`로 주입한다.
- `run.sh`는 `data/prices.db`와 리뷰 색인이 없을 때만 만든 뒤 FastAPI(8000) → Streamlit(8501)을 띄운다. 호스트 `data/`를 마운트하면 기존 DB·색인을 그대로 쓰고, 빈 볼륨이면 시작 시 생성한다(약 15초 추가).
- 헬스체크는 `GET /health`. 모델 예열이 끝나면 healthy가 된다(약 20초). `docker ps`의 STATUS에서 확인.

### 2-B. 로컬에서 실행
```bash
pip install --no-deps -r requirements.txt   # 전체 의존성 고정 목록 (사유는 requirements.txt 상단 주석)
bash run.sh        # 가격 DB·리뷰 인덱싱(없을 때만) → FastAPI(8000) → Streamlit(8501) 순서로 실행
```

Windows(Git Bash)에서는 가상환경의 Python을 지정해 실행한다: `PYTHON=.venv/Scripts/python bash run.sh`
Windows에서 새 가상환경을 프로젝트 폴더 밖의 깊은 경로에 만들면 torch 설치가 경로 길이(260자) 제한에 걸릴 수 있으니 프로젝트 폴더 안(`.venv`)에 만든다.

### 3. 사용
- UI: http://localhost:8501
- API 예시 (표준 규약):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "원룸에서 쓸 35만원 이하 무선청소기 추천해줘. 소음이 제일 중요해"}'
```
- 응답 형식:
```json
{
  "answer": "근거 기반 응답 (Top 3 모델 추천, 판매처별 실결제가, 리뷰 근거, 출처 링크)",
  "contexts": [{"doc_id": "blog-223456789#c3", "text": "리뷰 청크 원문"}],
  "trace": [{"step": "retrieve_reviews", "input": "확장 쿼리", "output": "리뷰 청크 8개"}]
}
```
  - `trace[].step`은 도구 호출 단계에서 도구 이름(`search_products`, `fetch_reviews`, `retrieve_reviews`, `query_price_db`, `calc_real_price`)을 그대로 쓰며, `test_queries.csv`의 `expected_tools`와 대조한다.
- 헬스체크: `GET http://localhost:8000/health` → `{"status": "ok"}` (규약 외 보조 엔드포인트, Docker 헬스체크용)

### 4. 평가 실행
```bash
python src/evaluate.py --queries evaluation/test_queries.csv --round 1
```

## RAGAS 평가 결과
<!-- TODO: 2차(최종) 평가 결과로 채우세요. 목표치는 괄호 안 -->
- context_recall: 0.XX (목표 0.75 이상)
- context_precision: 0.XX (목표 0.75 이상)
- faithfulness: 0.XX (목표 0.85 이상)
- answer_relevancy: 0.XX (목표 0.80 이상)

측정 방식: positive·edge 19문항 중 RAG를 거친(contexts가 있는) 문항 대상, `retrieve_reviews` 결과를 retrieved_contexts로 사용. 스키마에 정답 컬럼이 없어 context_recall의 reference는 `expected_traits`를 문장으로 이어 붙여 대신함.

1·2차 상세 결과: `evaluation/round1_report.md`, `evaluation/round2_report.md`

## 인-아웃 세트 통과율 (자체 평가)
<!-- TODO: 실제 결과로 채우세요. 완성 기준은 30문항 중 24문항(80%) 이상, guardrail 항목 100%
     통과 = expected_traits 모두 충족 + forbidden 미포함 + expected_tools 모두 호출
     카테고리 구성: positive 12 / negative 6 / edge 7 / guardrail 5 (권장 비율 40/20/25/15%)
     guardrail 5문항은 SERVICE.md 정책 5건 준수 여부, 인젝션 20개 차단율(95%)은 data/labels/로 별도 측정 -->
- 1차 (Day 9 종료): XX / 30 통과 — 벡터 단독 검색 기준선
- 2차 (Day 10 개선 후): XX / 30 통과 — 하이브리드 + 리랭킹 + 쿼리 확장 + 협찬 청크 제외, 가드레일 보완
- 개선폭: +XX 건 (주요 개선 사항: ...)

## 트라이앤에러 회고
<!-- TODO: 실제 경험으로 바꾸세요. 아래는 이 설계에서 나올 법한 항목 예시 -->
- **시도했지만 실패한 접근 · 왜 실패했는가**
  - 예: 벡터 검색만으로 리뷰를 찾았더니 모델명·숫자(흡입력 W, 소음 dB)가 들어간 질문에서 관련 리뷰를 놓쳤다.
  - 예: 협찬 판별을 정규식 문구만으로 했더니 문구 없이 쓴 광고성 리뷰를 걸러내지 못했다.
- **최종 채택한 접근 · 왜 그것으로 갔는가**
  - 예: BM25 + 벡터 하이브리드에 리랭킹을 더해 키워드 정확도와 의미 검색을 함께 확보했다.
  - 예: 문구 규칙으로 1차 거르고 LLM 판별로 2차 확인, 판단이 어려운 리뷰는 "판별 불가"로 따로 표시했다.
- **남은 한계 · 향후 개선 방향**
  - 상품·가격이 샘플 카탈로그라 실제 시세를 반영하지 못한다 (네이버 쇼핑 검색 API가 2026-07-31 대체 API 없이 종료) → `search_products` 인터페이스를 유지했으므로 공식 가격 API 확보 시 데이터 소스만 교체.
  - 블로그 검색 API는 요약 스니펫만 제공해 실데이터 모드의 협찬 판별 근거가 약하고, 상품명이 가상이라 블로그 결과는 상품 근거가 아닌 참고 후기로만 쓴다 → 쇼핑몰 구매평 등 본문 제공 API 연동.
  - 예: 가격 이력이 짧아 구매 타이밍 판단이 어렵다 → 장기 수집과 가격 알림(HITL 승인) 추가.

## 핵심 코드 위치
<!-- TODO: 구현 후 실제 줄 번호(NN)로 바꾸세요 -->
- `src/agent.py:NN` — 메인 에이전트 그래프 (ReAct)
- `src/tools.py` — 도메인 도구 5종
- `src/retriever.py` — RAG 파이프라인 (하이브리드 검색·리랭킹·쿼리 확장·협찬 청크 제외)
- `src/chains.py:NN` — LCEL 체인과 Pydantic 스키마
- `src/guardrails.py:NN` — 인젝션 탐지·PII 마스킹
- `src/api.py:NN` — FastAPI 엔드포인트
- `src/evaluate.py:NN` — RAGAS·LLM-as-Judge 평가
