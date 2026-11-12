# 미니 PJT: 진짜후기 쇼핑 비교 에이전트

## 무엇을 푸나
최저가와 리뷰는 넘쳐나지만 협찬 리뷰와 결제 조건 때문에 "나한테 진짜 맞는 상품"을 고르기 어려운 문제를, 협찬 리뷰를 걸러낸 실구매 후기와 내 조건 기준 실결제가로 서로 다른 모델 상위 3개를 추천하고 모델마다 판매처별 실결제가를 비교해 해결한다.

## 활용한 패턴 (Day 1~12)
- Day 1: LCEL chain (Pydantic 구조화 출력) — 리뷰 분석·추천 카드를 `prompt | llm.with_structured_output(ReviewAnalysis)` 체인으로 생성하고, 같은 Pydantic 모델을 API 응답 스키마로 재사용
- Day 3: ReAct (도구 자율 선택) — 에이전트가 질문에 따라 상품 검색·리뷰 검색·가격 계산 도구를 스스로 골라 호출 (요청당 최대 6회). 에이전트가 리뷰 검색·실결제가 중 하나를 빠뜨리면 코드가 보충 호출하고 trace에 "(자동 보충)"으로 남긴다
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
RETRIEVER_MODE=hybrid          # hybrid: BM25+벡터+리랭킹+협찬 제외 (최종·2차, 기본값) / vector: 1차 기준선
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
API_URL=http://localhost:8000  # Streamlit UI가 호출할 API 주소 (UI와 API를 따로 띄울 때 지정)
API_TIMEOUT=120                # Streamlit UI의 API 요청 제한 시간(초)
```

`.env`는 `docker run --env-file`로 그대로 읽히므로 값 뒤에 주석이나 따옴표를 붙이지 않는다(위 예시의 `#` 주석은 설명용).

### 2-A. Docker로 실행 (클린 환경 재현)
```bash
docker build -t mini-pjt-shopping .
docker run --env-file .env -p 8000:8000 -p 8501:8501 \
  -v "$(pwd)/data:/app/data" mini-pjt-shopping   # Chroma 인덱스·SQLite를 호스트에 보존 (이미지 WORKDIR은 /app)
```

Windows Git Bash에서는 경로 자동 변환 때문에 위 명령의 볼륨이 `/app/data`에 붙지 않는다(컨테이너가 데이터를 새로 만들고 호스트에 남지 않음). 경로 변환을 끄고 Windows 경로를 넘긴다.
```bash
MSYS_NO_PATHCONV=1 docker run --env-file .env -p 8000:8000 -p 8501:8501 \
  -v "$(pwd -W)/data:/app/data" mini-pjt-shopping
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
python3.14 -m venv .venv && source .venv/bin/activate
pip install --no-deps -r requirements.txt   # 전체 의존성 고정 목록 (사유는 requirements.txt 상단 주석)
bash run.sh        # 가격 DB·리뷰 인덱싱(없을 때만) → FastAPI(8000) → Streamlit(8501) 순서로 실행
```

Windows PowerShell (run.sh는 Git for Windows의 bash로 실행한다):
```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install --no-deps -r requirements.txt
$env:PYTHON = ".venv/Scripts/python"; bash run.sh
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
Windows PowerShell (한글이 깨지지 않게 본문을 UTF-8 바이트로 보낸다):
```powershell
$body = @{ question = "원룸에서 쓸 35만원 이하 무선청소기 추천해줘. 소음이 제일 중요해" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
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
python src/evaluate.py --queries evaluation/test_queries.csv --round 1   # 1차: RETRIEVER_MODE=vector로 API를 띄워 측정
python src/evaluate.py --queries evaluation/test_queries.csv --round 2   # 2차: RETRIEVER_MODE=hybrid
```
- 평가 스크립트가 라운드에 맞는 설정으로 API 서버를 직접 띄우고(`REVIEW_SOURCE=sample`), 30문항 → LLM-as-Judge → RAGAS → 협찬 판별 → 인젝션 차단율 순서로 측정해 `evaluation/round{N}_report.md`를 쓴다.
- 문항별 결과를 `evaluation/round{N}_results.json`에 저장하므로 Bedrock 일일 한도 등으로 멈추면 같은 명령으로 이어서 실행한다. `--fresh`는 처음부터 다시, `--rescore-ragas`는 저장된 응답으로 RAGAS만 다시, `--report-only`는 리포트만 다시 쓴다.

## RAGAS 평가 결과
2차(최종, hybrid) 결과. 괄호 안은 목표, 화살표 앞은 1차(vector 기준선).
- context_recall: **0.88** (목표 0.75 이상) — 1차 0.77 → 달성
- context_precision: **0.79** (목표 0.75 이상) — 1차 0.62 → 달성
- faithfulness: **0.75** (목표 0.85 이상) — 1차 0.71 → **미달**
- answer_relevancy: **0.86** (목표 0.80 이상) — 1차 0.75 → 달성

그 밖의 성공 지표(2차): LLM-as-Judge 평균 4.90(목표 4.0), 협찬 판별 정확도 100%(50/50, 숨은 광고 4/4), 인젝션 차단율 100%(20/20, 일반 리뷰 154건 중 오탐 0건), 응답 시간 최대 20.0초·평균 9.7초(목표 30초). 8개 중 faithfulness를 뺀 7개를 달성했다.

faithfulness 미달 원인: 답변 중 리뷰 문장의 "근거 없음" 판정은 대부분 리뷰 사실을 사용자 조건으로 옮겨 쓴 연결 문장("4인 가족에게 충분한 용량", "수면 중 사용에 부적합")이다. 리뷰 표현을 살리라는 프롬프트 규칙을 넣고 다시 쟀지만 평균은 그대로였다(문항별로 오르내림). 상세는 `evaluation/round2_report.md` "남은 한계".

측정 방식: positive·edge 19문항 중 RAG를 거친(contexts가 있는) 14문항 대상, `retrieve_reviews` 결과를 retrieved_contexts로 사용(청크 앞에 상품명). 판정은 **리뷰 범위 정의**로 한다: faithfulness는 답변 중 리뷰로 만든 문장만, context_precision·recall의 reference는 `expected_traits` 중 리뷰로 답할 수 있는 항목만 이어 붙여 쓰고, answer_relevancy는 답변 전체를 쓴다. 가격·판매처는 도구 결과라 RAGAS 대신 출력 가드레일 금액 검사와 Judge가 확인한다. 처음 정의(답변 전체·기대 항목 전체)의 수치도 리포트에 함께 싣는다 (변경 이유는 PRD "RAGAS 연결"). 처음 정의로는 2차 faithfulness 0.24, context_precision 0.32, context_recall 0.45, answer_relevancy 0.85다.

1·2차 상세 결과: `evaluation/round1_report.md`, `evaluation/round2_report.md`

## 인-아웃 세트 통과율 (자체 평가)
완성 기준은 30문항 중 24문항(80%) 이상 + guardrail 5문항 전부 통과. 통과 = `expected_traits` 모두 충족(Judge) + `forbidden` 미포함(규칙·Judge) + `expected_tools` 모두 호출(trace). 인젝션 20개 차단율은 `data/labels/`로 따로 잰다.

| 구분 | positive (12) | negative (6) | edge (7) | guardrail (5) | 합계 |
| --- | --- | --- | --- | --- | --- |
| 1차 (Day 9, vector 기준선) | 6 | 3 | 3 | 5 | **17 / 30** (미달) |
| 2차 (Day 10, hybrid + 개선) | 12 | 6 | 7 | 5 | **30 / 30** (달성) |

- 개선폭: **+13건**. 주요 개선 사항:
  - 리뷰 검색·실결제가 자동 보충(A1·A1′): 도구 누락으로 인한 실패 해소
  - hybrid 검색(협찬 청크 제외·모델별 실구매 우선 할당)
  - 여러 구조화 출력 중 가장 완전한 것 선택(A2)
  - 후기 요약에 판매처별 가격(A4), 근거 없는 조건 "확인 불가" 표시(A9)
  - 실결제가 우선 질문의 가격 가중치·카테고리 인접 상품 후순위
- 2차는 같은 설정으로 두 번 돌렸다. 첫 완료 실행은 28/30이었고, 남은 실패 #12(순위)를 고친 뒤 처음부터 다시 돌려 30/30이 나왔다. 다른 실패 #3은 Judge 오판이라 코드를 바꾸지 않았는데 재실행에서 통과했다. Judge 판정에 ±1문항 정도 편차가 있다고 본다.

## 트라이앤에러 회고
근거: git log(단계별 커밋), `evaluation/round1_report.md`, `evaluation/round2_report.md`.

- **시도했지만 실패한 접근 · 왜 실패했는가**
  - **벡터 단독 검색(1차 기준선).** 협찬 글은 조건어를 고르게 담은 홍보 문장이라 검색 점수가 높아 Top k를 차지했고, 실구매 리뷰가 하나뿐인 모델은 근거 없이 최저가만으로 상위 카드가 됐다. 1차 Context Precision 0.62.
  - **도구 선택을 LLM에만 맡기기.** 에이전트가 필요한 도구 셋(상품 검색·리뷰 검색·실결제가) 중 하나를 그때그때 다르게 빠뜨렸다. 1차는 `retrieve_reviews` 누락 4건이었고, 그것을 고친 2차 첫 실행에서는 이번엔 `calc_real_price` 누락 10건으로 20/30이 나왔다. 프롬프트 규칙만으로는 막지 못했다.
  - **카드 문구를 따로 만드는 LLM 호출.** 리뷰 분석 뒤 카드 문구 호출을 하나 더 두었더니 응답 시간이 30초를 넘었다(6단계).
  - **구조화 출력의 첫 도구 호출만 쓰기.** 모델이 한 응답에 초안과 완성본을 함께 내는 경우가 있어, 실구매 리뷰가 있는데도 "실구매 후기 없음" 카드가 나왔다(1차 A2).
  - **RAGAS를 답변 전체로 재기(원 정의).** 답변 대부분이 도구가 계산한 가격·판매처 줄이라 리뷰 청크로는 근거를 찾을 수 없어 faithfulness 0.18이 나왔다. 리뷰가 답할 수 없는 부분까지 리뷰 근거로 재고 있었다.
  - **리뷰 표현 충실 프롬프트 규칙.** faithfulness를 올리려고 리뷰 문구를 살리고 단정·과장을 금지했지만 평균 0.75 그대로였다. 남은 근거 없음은 과장이 아니라 리뷰 사실을 사용자 조건에 잇는 문장에서 나왔다.
  - **가격 점수를 카테고리 구분 없이 적용.** 음식물처리기 질문에서 처리 기능이 없는 냉장 보관통이 가장 싸다는 이유로 1위가 되고, 질문이 기대한 한결카드 최저가 모델이 4위로 밀렸다(2차 #12).
- **최종 채택한 접근 · 왜 그것으로 갔는가**
  - **hybrid 검색 + 협찬 청크 제외 + 모델별 실구매 우선 할당.** 대가성 문구가 있는 리뷰 청크를 검색 후보에서 빼고 모델마다 실구매 청크부터 채웠다. 모델별 할당 때문에 4단계 측정에서 Precision@8이 0.93→0.80으로 떨어지는 트레이드오프가 있었지만, 카드마다 실구매 근거를 확보하는 쪽을 택했다. 2차 Context Precision 0.79, Recall 0.88.
  - **도구 자동 보충(A1·A1′).** 리뷰 수집 뒤 검색이 없으면 `retrieve_reviews`를, 리뷰 검색 뒤 실결제가가 없고 가격·추천 의도이면 `calc_real_price`를 코드가 보충한다. 카드·멤버십은 카탈로그에 있는 이름이 질문에 나오는지로 찾아 LLM이 옮겨 적지 않게 했다. 응답 경로가 안정되고 LLM 턴이 줄었다(2차 30문항 중 18문항에서 동작).
  - **리뷰 분석 1회에 협찬 판별·장단점·카드 문구 통합.** 모델당 한 번의 구조화 출력 호출(병렬)로 받고 순위·실결제가·카드 조립은 코드가 한다. 2차 최대 응답 시간 20.0초.
  - **협찬 판별은 문구 규칙 → LLM 2단계.** 규칙으로 대가성 문구를 먼저 거르고 LLM이 숨은 광고를 판별한다. 50건 라벨 기준 100%, 문구 없는 숨은 광고 4/4.
  - **RAGAS 리뷰 범위 정의.** faithfulness는 답변 중 리뷰로 만든 문장만, precision·recall의 reference는 리뷰로 답할 수 있는 기대 항목만 쓴다. 가격은 출력 가드레일 금액 검사와 Judge가 본다. 원 정의 수치도 리포트에 함께 남겼다.
  - **실결제가 우선 가중치 + 인접 상품 후순위.** "실결제가 기준·최저가" 질문은 가격 가중치를 0.5로 올리고, 상품명에 카테고리가 없는 인접 상품은 같은 카테고리 상품 뒤에 둔다. #12 통과.
  - **Bedrock 일일 토큰 한도 대응.** 일일 한도는 재시도하지 않고 바로 503(Retry-After)을 돌려주고, 평가는 문항별로 저장해 이어서 실행한다. 같은 모델이라도 `us.`·`global.` 추론 프로필은 한도가 따로라 평가 중 소진 시 프로필을 바꿔 이어 갔다(리포트에 경위 기록).
- **남은 한계 · 향후 개선 방향**
  - **faithfulness 0.75 (목표 0.85 미달).** 추천 근거를 "리뷰 인용 + 대응 조건" 형식으로 바꿔 판단 문장을 줄이거나, 조건 연결 문장을 RAGAS 대상에서 빼는 정의 변경이 남은 방법이다. 전자는 가독성, 후자는 지표 의미가 바뀌어 반영하지 않았다.
  - **ReAct 자율성 축소.** 2차 30문항 중 18문항에서 코드가 도구를 보충했다. 안정성을 얻은 대신 에이전트의 도구 자율 선택 범위가 줄었다 → 도구 설명·예시를 다듬어 보충 비율을 낮추는 것이 다음 과제.
  - **Judge 판정 흔들림.** 같은 수준의 답변을 Judge가 3점과 5점으로 다르게 채점한 사례(#3)가 있다 → 여러 번 채점해 다수결로 판정.
  - 상품·가격이 샘플 카탈로그라 실제 시세를 반영하지 못한다 (네이버 쇼핑 검색 API가 2026-07-31 대체 API 없이 종료) → `search_products` 인터페이스를 유지했으므로 공식 가격 API 확보 시 데이터 소스만 교체.
  - 블로그 검색 API는 요약 스니펫만 제공해 실데이터 모드의 협찬 판별 근거가 약하고, 상품명이 가상이라 블로그 결과는 상품 근거가 아닌 참고 후기로만 쓴다 → 쇼핑몰 구매평 등 본문 제공 API 연동.
  - 후보 모델 5개 리뷰 분석이 동시 호출 상한 3에 걸려 두 번에 나눠 실행된다 → 카드에 오를 상위 모델만 분석해 응답 시간 여유 확보.

## 핵심 코드 위치
- `src/agent.py:555` — 메인 에이전트 그래프 (ReAct, `build_graph`). 도구 실행·자동 보충 `tools_node` 285, 진입점 `run` 635
- `src/tools.py:260` — 도메인 도구 5종 (`search_products` 260, `fetch_reviews` 470, `retrieve_reviews` 534, `query_price_db` 605, `calc_real_price` 748)
- `src/retriever.py:272` — RAG 파이프라인 `search` (하이브리드 검색·리랭킹·쿼리 확장·협찬 청크 제외). 쿼리 확장 `expand_queries` 131, 리랭커 `CrossEncoderReranker` 171
- `src/chains.py:105` — LCEL 체인과 Pydantic 스키마 (`ReviewAnalysis` 105, `RecommendationCard` 168, 리뷰 분석 체인 `review_analysis_chain` 512, 순위·카드 조립 `build_cards` 790)
- `src/guardrails.py:163` — 인젝션 탐지 `detect_injection` (PII 마스킹 `mask_pii` 58, 리뷰 정화 `sanitize_review` 133, 입력 검사 `check_input` 281, 출력 검사 `check_output` 337)
- `src/api.py:130` — FastAPI 엔드포인트 `POST /query` (`GET /health` 109)
- `src/evaluate.py:458` — RAGAS 평가 `run_ragas` (LLM-as-Judge `run_judge` 266, 리포트 `build_report` 539)
