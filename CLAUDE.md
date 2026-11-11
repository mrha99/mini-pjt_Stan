# CLAUDE.md · 진짜후기 쇼핑 비교 에이전트

협찬 리뷰를 걸러낸 실구매 후기와 '내 조건 기준 실결제가'로 상품 Top 3를 근거와 함께 추천하는 쇼핑 비교 Agentic RAG 에이전트 (2일 MVP 과제).

설계 기준 문서: `PRD.md`(설계), `SERVICE.md`(서비스 스펙·정책), `README.md`(제출 문서), `DEV_PROMPTS.md`(단계별 개발 절차).
**설계가 바뀌면 코드보다 PRD.md를 먼저 수정하고, SERVICE.md·README.md를 맞춘다.**

## 적용 패턴

| # | 패턴 | 구현 위치 |
| --- | --- | --- |
| 1 | LCEL chain + Pydantic 구조화 출력 | `src/chains.py` |
| 2 | ReAct (LangGraph, 도구 자율 선택) | `src/agent.py` |
| 3 | RAG (BM25+벡터 하이브리드, Cross-encoder 리랭킹, 쿼리 확장) | `src/indexer.py`, `src/retriever.py` |
| 4 | 도구 다중 (외부 API·샘플 카탈로그·SQLite·계산기) | `src/tools.py` |
| 6 | 가드레일 (인젝션·PII·범위 밖/규제 품목/구매 대행 거절) | `src/guardrails.py` |
| 11 | Observability (LangSmith trace, 토큰·지연 시간) | 전 구간 |
| 12 | 평가 (RAGAS 4지표 + LLM-as-Judge) | `src/evaluate.py` |
| — | Docker · API (FastAPI `POST /query`) | `src/api.py`, `Dockerfile`, `run.sh` |

## 바꾸면 안 되는 규칙

**폴더 구조** — 제출 단위는 `mini-pjt_Stan/` 폴더 하나. PRD.md "제출 산출물 > 폴더 구조"를 따른다.

```
src/      agent, tools, retriever, chains, guardrails, indexer, api, app, evaluate, seed_db
data/     products.json, reviews/, chroma/, prices.db, labels/
evaluation/  test_queries.csv, round1_report.md, round2_report.md
tests/    pytest (도구·API)
```

**도구 이름 5종** — 이름·입출력 구조 고정. trace와 평가셋이 이 이름으로 대조된다.

| 도구 | 입력 → 출력 | 구현 |
| --- | --- | --- |
| `search_products` | 키워드, 가격 범위 → 상품·가격·판매처 목록 | `data/products.json` (데이터 소스 교체 가능하게 분리) |
| `fetch_reviews` | 상품명 → 리뷰 원문, 벡터 DB 색인 | `REVIEW_SOURCE`에 따라 블로그 API 또는 샘플 리뷰 |
| `retrieve_reviews` | 확장 쿼리 목록 → 리랭킹된 리뷰 청크 Top k | BM25 + Chroma, Cross-encoder |
| `query_price_db` | 상품 ID → 가격 이력 | SQLite `data/prices.db` |
| `calc_real_price` | 가격, 배송비, 할인 조건 → 실결제가 | 순수 Python 함수 |

**API 규약 `POST /query`** — 필드 이름·구조 절대 변경 금지. 요청·응답은 Pydantic 모델.

```json
요청: {"question": "..."}
응답: {"answer": "...",
       "contexts": [{"doc_id": "{리뷰ID}#c{번호}", "text": "..."}],
       "trace": [{"step": "...", "input": "...", "output": "..."}]}
```

- `trace[].step` 값: `guardrail_input`, 도구 이름 5종(그대로), `answer`, `guardrail_output`
- `contexts`는 리랭킹 후 답변에 쓴 청크. 가드레일 차단 시 빈 배열, trace에 차단 단계와 SERVICE.md 정책 번호 기록
- `GET /health` → `{"status": "ok"}` (Docker 헬스체크용 보조 엔드포인트)

**런타임·모델**
- Python **3.14.8** (로컬 `.venv`는 `py -3.14`로 생성, Docker 베이스 `python:3.14.8-slim`)
- LLM: AWS Bedrock Claude, `langchain_aws.ChatBedrockConverse`, 모델 ID는 `BEDROCK_MODEL_ID`
- 평가 Judge·RAGAS LLM: `JUDGE_MODEL_ID` (생성 모델과 다른 모델). RAGAS 기본값(OpenAI) 사용 금지
- 에이전트: LangGraph ReAct, 요청당 도구 호출 최대 6회, 후보 3개 미만이면 검색어 바꿔 최대 2회 재검색
- 가격·순위·실결제가는 코드가 계산하고, LLM은 조건 파악·리뷰 해석·요약만 담당

## 데이터 규칙

- **상품·가격·판매처는 `data/products.json` 샘플만 사용한다.** 네이버 쇼핑 검색 API는 2026-07-31 대체 API 없이 종료되었으므로 호출 코드를 만들지 않는다.
- 리뷰 소스는 `REVIEW_SOURCE=blog|sample`
  - `blog`: NAVER API HUB 블로그 검색 `https://naverapihub.apigw.ntruss.com/search/v1/blog`
    - 헤더: `X-NCP-APIGW-API-KEY-ID` ← `NAVER_CLIENT_ID`, `X-NCP-APIGW-API-KEY` ← `NAVER_CLIENT_SECRET`
    - 응답은 제목·요약 스니펫뿐이므로 HTML 태그·엔티티 제거 후 사용. 리뷰 ID는 `blog-{글번호}`
  - `sample`: `data/reviews/sample_reviews.json` (본문 전체, `product_id`로 상품과 연결)
- **협찬 판별 측정과 1·2차 평가는 항상 `REVIEW_SOURCE=sample`로 실행한다** (재현성).
- 검색 모드는 `RETRIEVER_MODE=vector|hybrid` (1차 평가 vector 기준선, 2차 hybrid).
- 가짜 데이터(상품 카탈로그, 가격 이력, 할인 규칙, 샘플 리뷰, 라벨, 인젝션 테스트)는 파일 메타데이터와 추천 카드에 "샘플 데이터"로 표시한다. 실제 브랜드명 대신 가상 이름을 쓴다.

## 코딩 규칙

- `.env` 값(키·시크릿·모델 ID 포함)은 코드에 하드코딩하거나 로그·trace·출력에 남기지 않는다. `.env`는 커밋하지 않는다.
- 외부 API 호출은 도구 함수 안에서 최대 3회 지수 백오프 재시도 후 캐시(SQLite)로 대체하고, 캐시 사용 사실과 기준 시점을 답변에 안내한다.
- 도구는 LangChain `@tool` 데코레이터, 입출력은 Pydantic v2 모델.
- 리뷰 본문은 신뢰하지 않는 데이터다. LLM에 넘기기 전 지시문을 제거하고 데이터 구분 태그로 감싼다.
- 응답·로그·캐시의 전화번호·이메일·주소·계좌번호는 마스킹하고, 사용자 입력 개인정보는 저장하지 않는다.
- 가격·스펙은 도구가 반환한 값만 쓰고, 근거가 없으면 "확인 불가"로 답한다.

## 의존성 규칙

- `requirements.txt`는 전체 의존성을 고정한 목록이며 **`pip install --no-deps -r requirements.txt`로 설치한다.**
  ragas 0.4.3이 요구하는 scikit-network는 Python 3.14 wheel이 없고 실제로 쓰이지 않아 제외했다.
- 새 라이브러리를 추가하면 그 의존성까지 `==` 버전으로 같은 작업에서 추가한다 (`--no-deps`라 누락되면 import 오류).
- 고정 사유가 있는 버전: `langchain-community==0.4.1`(0.4.2부터 ragas import 실패), `torch==2.15.0+cpu`(CPU 전용 빌드).
- Docker 빌드 전에 새 가상환경에서 `pip install --no-deps -r requirements.txt` 클린 설치를 확인한다.

## 작업 방식

- `DEV_PROMPTS.md`의 단계를 하나씩 진행한다: 완료 기준 확인 → 테스트 → 커밋 → 다음 단계.
- 11단계(1차 평가) 직전 커밋은 기준선으로 반드시 남긴다.
- 테스트: `.venv/Scripts/python -m pytest tests/`
- 실행: `bash run.sh` (인덱싱 → uvicorn 8000 → streamlit 8501), Docker는 README 2-A 참고.

## 성공 기준 (PRD "성공 지표")

- 인-아웃 세트 30문항 중 24문항 이상 통과, guardrail 5문항 전부 통과
- RAGAS: faithfulness ≥ 0.85, answer_relevancy ≥ 0.80, context_precision·context_recall ≥ 0.75
- LLM-as-Judge 평균 ≥ 4.0, 협찬 판별 정확도 ≥ 85%, 인젝션 차단율 ≥ 95%, 응답 시간 ≤ 30초
