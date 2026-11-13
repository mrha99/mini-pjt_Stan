# TASKS · 진짜후기 쇼핑 비교 에이전트

PRD.md, SERVICE.md, README.md를 기준으로 한 단계별 개발 절차와 AI 코딩 도구(Claude Code 등)용 프롬프트 모음.

> **전제:** 네이버 쇼핑 검색 API가 2026-07-31 대체 API 없이 종료되어, 상품·가격은 `data/products.json` 샘플 데이터를 사용한다. 리뷰는 설정값 `REVIEW_SOURCE=blog|sample`로 블로그 API(NAVER API HUB) 또는 샘플 리뷰를 선택한다. 블로그 API는 요약 스니펫만 주므로 협찬 판별과 1·2차 평가는 `REVIEW_SOURCE=sample`로 실행한다. 데이터 방향이 바뀌면 0·1·3단계 프롬프트를 수정한다.

## 프롬프트 작성 원칙

- 모든 프롬프트는 `@PRD.md` 등 문서를 참조해 설계 기준을 반복 설명하지 않는다.
- 한 번에 한 단계만 요청하고, 완료 기준 확인 → 테스트 → 커밋 후 다음 단계로 넘어간다.
- 바꾸면 안 되는 것(도구 이름, API 규약, 폴더 구조)을 프롬프트에 명시한다.
- 단계마다 새 세션을 열고, 공통 맥락은 `CLAUDE.md`로 전달한다.
- 새 라이브러리를 쓰는 단계에서는 `requirements.txt`에 버전을 고정해 함께 추가한다 (9단계 Docker 빌드와 제출 산출물의 기준).

## 단계 요약

| 단계 | 내용 | 산출물 | 시점 |
| --- | --- | --- | --- |
| 0 | 프로젝트 컨텍스트·환경 점검 | CLAUDE.md, .venv, requirements.txt | 착수 |
| 1 | 샘플 데이터 준비 | data/ | 착수 |
| 2 | 평가셋 작성 | evaluation/test_queries.csv | 착수 |
| 3 | 도구 5종 | src/tools.py | 1일차 |
| 4 | RAG 파이프라인 | src/indexer.py, src/retriever.py | 1일차 |
| 5 | LCEL 체인 | src/chains.py | 1일차 |
| 6 | ReAct 에이전트 + Trace | src/agent.py | 1일차 |
| 7 | 가드레일 | src/guardrails.py | 1일차 |
| 8 | API | src/api.py | 2일차 |
| 9 | Docker·run.sh | Dockerfile, run.sh | 2일차 |
| 10 | UI | src/app.py | 2일차 |
| 11 | 1차 평가 | src/evaluate.py, round1_report.md | Day 9 |
| 12 | 개선·2차 평가 | round2_report.md | Day 10 |
| 13 | 문서 마무리·최종 검증 | README.md, SERVICE.md | Day 10 |

---

## 0단계: 프로젝트 컨텍스트 설정과 환경 점검

```
@PRD.md @SERVICE.md @README.md 를 읽고 이 프로젝트의 CLAUDE.md를 작성해줘.
포함할 내용:
- 프로젝트 목적 한 줄, 적용 패턴 7개(1·2·3·4·6·11·12)와 Docker·API
- 바꾸면 안 되는 규칙: 폴더 구조(mini-pjt_Stan), 도구 이름 5종, POST /query 규약,
  trace step = 도구 이름, Python 3.14.8, LLM은 AWS Bedrock Claude(ChatBedrockConverse)
- 데이터 규칙: 상품·가격은 data/products.json만 사용(쇼핑 검색 API 호출 금지, 2026-07-31 종료),
  리뷰는 REVIEW_SOURCE=blog|sample로 선택하고 평가는 sample로 실행,
  네이버 키(NAVER_CLIENT_ID/SECRET)는 NAVER API HUB 블로그 검색용
- 코딩 규칙: .env 값은 코드·로그에 출력 금지, 외부 API는 3회 재시도 후 캐시 대체
- 설계가 바뀌면 PRD.md를 먼저 수정할 것
- 의존성 규칙: 새 라이브러리는 requirements.txt에 == 버전으로 고정해 추가
CLAUDE.md는 200줄 이내로 작성해줘.

이어서 개발 환경을 점검해줘.
1. py -3.14 로 .venv 가상환경을 만들고 Python 3.14.8인지 확인
2. 핵심 라이브러리(langchain, langchain-aws, langgraph, chromadb, rank-bm25,
   sentence-transformers, ragas, fastapi, uvicorn, streamlit, langsmith, pytest, httpx)를
   설치해 3.14에서 설치·import 되는지 확인하고, 성공한 버전으로 requirements.txt를 고정
3. 설치가 안 되는 라이브러리가 있으면 대안을 제안 (진행은 내가 결정)
4. .env 값으로 Bedrock(BEDROCK_MODEL_ID, JUDGE_MODEL_ID) 1회 호출과 LangSmith 연결 확인 (키 값은 출력 금지)
5. docker --version 으로 Docker 설치 여부 확인
```

**완료 기준:** CLAUDE.md 생성, 3.14.8 가상환경에서 핵심 라이브러리 import 성공, requirements.txt 버전 고정, Bedrock·LangSmith 연결 확인, Docker 설치 확인 후 커밋

## 1단계: 샘플 데이터 준비

```
@PRD.md @SERVICE.md 기준으로 data/ 폴더의 샘플 데이터를 만들어줘.
1. data/products.json: 가전·생활용품 40개 (무선청소기, 공기청정기, 전기포트 등 5개 카테고리)
   필드: product_id, name, category, price, shipping_fee, mall, card_discounts, membership_discounts, link
   - 같은 모델을 여러 판매처가 파는 경우를 포함해 실결제가 비교가 의미 있게 만들 것
   - 파일 상단 메타데이터에 "가짜 데이터(과제용)"라고 표시
2. data/prices.db: products의 최근 30일 가격 이력을 SQLite로 생성하는 스크립트(src/seed_db.py)
3. data/reviews/sample_reviews.json: products.json 상품과 product_id로 연결된 본문 전체 리뷰
   - 상품당 3~6개, 협찬(대가성 문구 포함)·실구매·애매한 리뷰를 섞을 것
   - 원룸·소음·반려동물 등 조건어가 드러나는 내용을 포함해 RAG 검색이 의미 있게 만들 것
   - 필드: review_id, product_id, title, body, author, date, link, 메타데이터에 "가짜 데이터(과제용)"
4. data/labels/sponsored_labels.csv: 협찬 판별용 리뷰 50개 (협찬 20, 실구매 30, 라벨 포함)
   - sample_reviews.json의 리뷰에서 뽑아 review_id로 연결할 것
5. data/labels/injection_tests.csv: 인젝션 문구가 섞인 리뷰 20개
실제 브랜드명은 쓰지 말고 가상의 이름을 써줘.
```

**완료 기준:** 5개 파일 생성, `src/seed_db.py` 정상 실행

> 실행 메모: 이후 negative 평가용으로 리뷰 없는 상품 6개(KT-008, AP-008, KT-006, FW-006, VC-003, HM-003)를 추가해 실제 리뷰는 154개다. 목록은 `sample_reviews.json`의 `_meta.test_fixtures`에 있다.

## 2단계: 평가셋 작성

```
@PRD.md 의 "test_queries.csv 스키마"와 @SERVICE.md 4번 정책을 기준으로
evaluation/test_queries.csv 30문항을 만들어줘.
- 컬럼: id, category, input, expected_traits, forbidden, expected_tools, note (여러 값은 ;)
- 구성: positive 12, negative 6, edge 7, guardrail 5
- guardrail 5문항은 SERVICE.md 정책 1~5번과 하나씩 짝짓고 note에 정책 번호 표기
- expected_tools는 도구 이름 5종만 사용
- input과 기대 결과는 data/products.json의 실제 상품과 맞아야 함
- 리뷰 근거를 요구하는 expected_traits(예: 소음 관련 후기)는 data/reviews/sample_reviews.json에 실제로 있어야 함
- 데이터에 맞춘 작성 기준:
  - 결제 조건은 데이터의 가상 카드(한결·누리·새봄·다솜카드)와 멤버십(하늘몰 VIP·바로플러스·누리패스·온새 멤버스) 이름 사용
  - Top 3는 서로 다른 모델 3개 기준. positive "3개 추천" 문항은 예산 안에 서로 다른 모델이 3개 이상 있게 잡고
    (예: 무선청소기 35만원 이하), "30만원 이하 무선청소기"처럼 모델이 2개뿐인 조건은 edge(모델 3개 미만)로 사용
  - "리뷰 없는 상품" negative는 판매처가 한 곳뿐인 상품(KT-008, AP-008, KT-006, FW-006) 사용
    (VC-003·HM-003은 같은 모델의 다른 판매처에 리뷰가 있어 부적합)
  - "협찬 리뷰만 있는 상품" edge는 HM-008 사용
  - "근거 없는 스펙" 문항은 상품 specs에 없는 항목(예: 무선청소기 흡입력 W)으로 질문
  - guardrail 문항의 note에는 SERVICE.md 정책 번호를 하나만 표기
작성 후 카테고리별 문항 수와 정책 매핑을 표로 보여줘.
```

**완료 기준:** 30문항, 카테고리 비율·정책 매핑 일치

## 3단계: 도구 5종 구현

```
@PRD.md "도구 설계" 표대로 src/tools.py 를 구현해줘.
- search_products: data/products.json에서 키워드·가격 범위로 검색
  (나중에 실제 가격 API로 바꿀 수 있게 데이터 소스 부분을 분리하고, 도구 이름·입출력은 고정)
  - 출력의 각 상품에 product_id, model_id, 상품명, 판매처, 가격, 배송비, card_discounts, membership_discounts, 링크 포함
    (6단계에서 model_ids 전달·판매처 묶기에 사용)
- 도구 호출 최대 6회 안에 추천을 끝내기 위해 fetch_reviews와 calc_real_price는 여러 건을 한 번에 처리 (@PRD.md 도구 설계)
- fetch_reviews: 입력은 상품명 목록(+선택: 조건어). 설정값 REVIEW_SOURCE=blog|sample 에 따라
  - blog: NAVER API HUB 블로그 검색(naverapihub.apigw.ntruss.com/search/v1/blog),
    .env의 NAVER_CLIENT_ID → X-NCP-APIGW-API-KEY-ID, NAVER_CLIENT_SECRET → X-NCP-APIGW-API-KEY 헤더,
    응답은 제목·요약 스니펫이므로 HTML 태그(<b> 등)·엔티티 제거 후 사용.
    상품명이 가상이므로 상품명으로 카테고리를 찾아 "카테고리 + 조건어 + 후기"로 검색하고,
    결과는 "실제 블로그 참고 후기"로 표시해 특정 상품의 근거로 쓰지 않음 (@PRD.md 데이터 출처 변경 메모)
  - sample: 상품명마다 data/products.json에서 model_id를 찾고,
    data/reviews/sample_reviews.json에서 그 모델의 모든 판매처 리뷰를 모아 반환 (리뷰 근거는 모델 단위)
- retrieve_reviews: 4단계에서 만들 retriever를 호출하는 자리만 두기
- query_price_db: data/prices.db 조회
- calc_real_price: 순수 함수. 입력은 판매처별 상품 목록(각각 판매가, 배송비, card_discounts·membership_discounts)과
  사용자가 가진 카드·멤버십 목록(없으면 빈 목록), 출력은 상품별 실결제가 목록. 상품마다 아래 규칙 적용
  - 사용자가 가진 카드·멤버십만 적용, 카드는 할인액이 가장 큰 1장만 적용
  - 적용 순서: 멤버십 할인(free_shipping이면 배송비 0) → 카드 할인
  - 카드 할인액과 min_amount 판정은 배송비를 뺀 상품 금액(멤버십 적용 후) 기준, 실결제가 = 상품 금액 - 할인 + 배송비
  - rate/amount 구분, min_amount 미만이면 미적용, max_discount 상한 적용, 원 단위 절사
  - 출력: 실결제가와 적용 내역(어떤 할인이 얼마 적용됐는지)
규칙: LangChain @tool 데코레이터, 입출력은 Pydantic 모델, 외부 호출은 3회 지수 백오프 후 캐시.
캐시는 폴더 구조에 맞춰 data/prices.db 안의 별도 테이블에 둘 것 (seed_db.py는 자기 테이블만 다시 만들므로 캐시는 유지됨).
calc_real_price는 위 규칙별 경계값(최소 결제액 미달, 최대 할인 상한, 무료배송, 미보유 카드)을 테스트로 고정해줘.
도구별 pytest를 tests/test_tools.py에 작성하고 실행 결과를 보여줘.
blog 모드 테스트는 HTTP 호출을 mock으로 대체해 키 없이도 통과하게 해줘.
```

**완료 기준:** 5개 도구 단위 테스트 통과

## 4단계: RAG 파이프라인

```
@PRD.md 패턴 3번 기준으로 src/indexer.py, src/retriever.py 를 구현해줘.
- indexer: 리뷰를 청크로 나눠 Chroma(data/chroma/)에 색인, 청크 ID는 "{리뷰ID}#c{번호}"
  - 청크 메타데이터에 review_id, product_id, model_id, source(sample|blog), category를 넣어
    모델 단위로 검색·필터할 수 있게 할 것
    (리뷰 JSON에는 model_id가 없으므로 data/products.json에서 product_id로 찾아 넣을 것)
  - blog 리뷰는 특정 상품과 연결되지 않으므로 product_id·model_id 없이 source=blog, category만 넣는다.
    model_ids 필터 검색에서는 빠지며, 답변 근거(contexts)가 아니라 "실제 블로그 참고 후기"로 카드 아래에 따로 표시
  - 초기 색인(python -m src.indexer, run.sh에서 호출): data/reviews/sample_reviews.json 전체
  - blog 모드: fetch_reviews가 질의 때 받은 리뷰를 기존 색인에 추가(이미 있는 ID는 건너뜀),
    블로그 리뷰 ID는 "blog-{글번호}" (link의 마지막 숫자), 원문은 전화번호·이메일·주소·계좌번호를
    마스킹한 뒤 data/reviews/에 JSON으로 저장 (SERVICE.md 정책 3)
  - BM25 인덱스는 Chroma와 같은 청크 집합으로 만들고, 색인이 추가되면 함께 갱신
- retriever: 검색 모드를 설정값 RETRIEVER_MODE=vector|hybrid 로 선택
  - vector (1차 기준선): 쿼리 확장·리랭킹 없이 원 질의 1개로 Chroma만 사용
  - hybrid: 쿼리 확장 → BM25 + 벡터 앙상블 → 다국어 Cross-encoder 리랭킹 → 상위 k개
  - 쿼리 확장(hybrid만): 사용자 조건어(예: 원룸, 소음)를 붙인 멀티 쿼리 3개 생성
  - 두 모드 모두 선택 입력 model_ids(후보 모델 목록)가 있으면 그 모델의 청크만 검색 (@PRD.md 검색 범위)
  - 1·2차 평가를 공정하게 비교하도록 두 모드 모두 최종 반환 청크 수 k=8로 고정 (설정값으로 두기)
- 임베딩·리랭커는 한국어 지원 모델로 고르고 선택 이유를 주석으로 남겨줘
tools.py의 retrieve_reviews를 이 retriever에 연결하고,
같은 질의로 vector와 hybrid 결과를 비교 출력하는 기능도 만들어줘.
폴더 구조가 고정이므로 새 파일 대신 `python -m src.retriever --compare "질의"`로 실행되게 해줘.
```

**완료 기준:** 두 모드 동작, 비교 결과에서 hybrid의 관련 청크 검색 우위 확인

## 5단계: LCEL 체인

```
@PRD.md 패턴 1번 기준으로 src/chains.py 를 구현해줘.
- ReviewAnalysis 스키마 (모델 단위): 모델 ID, 리뷰별 협찬 판별 목록(review_id, 협찬/실구매/판별 불가, 판별 근거),
  실구매 리뷰 기준 조건별 장점·단점, 비추천 대상, 근거 청크 ID
- RecommendationCard 스키마: 모델명, 판매처별 실결제가 목록(판매처·가격·배송비·할인 반영 실결제가·링크,
  최저가 판매처 우선), 요약, 추천 근거, 출처 링크, 협찬 분리 표시,
  데이터 출처 표시(가격·판매처는 "샘플 데이터"로 렌더링)
- Top 3는 서로 다른 모델 3개 (같은 모델의 판매처들은 카드 하나에 묶음)
- 체인: prompt | llm.with_structured_output(스키마), LLM은 ChatBedrockConverse
- 협찬 판별은 규칙(대가성 문구)으로 1차, LLM으로 2차, 애매하면 "판별 불가"
  (규칙은 "협찬 아닙니다" 같은 부정 표현을 협찬으로 오판하지 않게 할 것)
- 추천 카드를 answer 텍스트로 렌더링하는 함수 포함
- LLM 호출은 지수 백오프 재시도와 동시 호출 수 제한을 적용 (Bedrock ThrottlingException 대비)
- 응답 시간 30초를 위해 추천 시 ReviewAnalysis는 모델당 1회 호출(그 모델의 리뷰 청크를 묶어서)로 처리하고,
  리뷰 단위 협찬 판별 LLM 호출은 규칙으로 판단이 안 되는 리뷰와 정확도 측정에만 사용
data/labels/sponsored_labels.csv로 협찬 판별 정확도를 측정해서 보여줘 (목표 85%).
정확도는 50개 전체를 분모로 하고 "판별 불가"는 오답으로 계산하며, 판별 불가 비율과
숨은 광고(label_type=hidden_ad) 4건의 결과는 따로 보여줘.
```

**완료 기준:** 스키마대로 구조화 출력, 협찬 판별 정확도 측정 완료

## 6단계: ReAct 에이전트 + Trace

```
@PRD.md 패턴 2·11번과 "추천 요청 처리 흐름"을 기준으로 src/agent.py 를 구현해줘.
- LangGraph ReAct 에이전트, 도구 5종 연결, 요청당 도구 호출 최대 6회
- 조건(예산·용도)이 없으면 되묻기, 조건에 맞는 모델이 3개 미만이면 검색어를 바꿔 최대 2회 재검색,
  그래도 부족하면 찾은 모델만 추천하고 그 사실을 안내
- 추천 1건의 도구 호출은 search_products 1회 → fetch_reviews 1회(후보 상품명 전체) → retrieve_reviews 1회
  → calc_real_price 1회(후보 판매처 전체)를 기본으로 하고, 재검색까지 최대 6회
- retrieve_reviews에는 search_products가 찾은 후보 모델의 model_ids를 함께 넘길 것
- 에이전트는 대화 상태를 저장하지 않는다. 되묻기는 질문을 answer로 돌려주고(trace step "answer") 끝내며,
  "이전 질문 + 되묻기 + 사용자 답"이 이어 붙은 question이 오면 하나의 요청으로 처리 (@PRD.md 되묻기 대화 이어가기)
- 최종 답변은 chains.py의 체인으로 생성
- 실행 결과에서 trace 리스트를 만들 것: step 값은 도구 이름 그대로, input·output은 요약,
  마지막에 체인으로 추천 카드를 만드는 단계는 step "answer"로 기록 (PRD API 스펙 예시와 동일)
- LangSmith 트레이싱이 켜져 있는지 확인하고, 요청별 총 토큰 수·지연 시간을 조회해 출력
CLI 실행 진입점을 만들고, REVIEW_SOURCE=sample 로 test_queries.csv의 positive 3문항을 돌려 결과와 trace,
문항별 응답 시간·토큰 수를 보여줘. 30초를 넘으면 가장 오래 걸린 단계를 짚고 줄일 방법을 제안해줘.
```

**완료 기준 (1일차 완료):** CLI에서 Top 3 추천 출력, LangSmith trace 기록, 응답 시간·토큰 수 확인

## 7단계: 가드레일

```
@SERVICE.md 4번 정책 5건과 @PRD.md 가드레일 설계 메모를 기준으로 src/guardrails.py 를 구현해줘.
- 입력: 인젝션 탐지(규칙+LLM 분류), 범위 밖·규제 품목·구매 대행 요청 분류 → 거절 + 할 수 있는 일 안내
- 리뷰: LLM에 넘기기 전 지시문 제거, 데이터 구분 태그로 감싸기
- 출력: 전화번호·이메일·주소·계좌번호 마스킹, 로그·캐시에도 마스킹 적용
- 차단 시 trace에 guardrail_input / guardrail_output 단계와 정책 번호 기록
agent.py 앞뒤에 연결하고, guardrail 5문항과 injection_tests.csv 20개를 돌려
정책별 통과 여부와 차단율(목표 95%)을 보여줘.
- guardrail 5문항: 사용자 질문이므로 에이전트 전체(입력 가드레일 → 에이전트 → 출력 가드레일)로 실행
- injection_tests.csv 20개: 사용자 질문이 아니라 리뷰 본문이므로 리뷰 정화 함수에 넣어 측정.
  지시문이 제거·무력화되면 차단 성공, expected_action=mask_and_remove(inj-20)는 PII 마스킹까지 확인.
  rv-0009·rv-0053처럼 코퍼스에 심어둔 인젝션 리뷰는 실제 질의로도 1건씩 돌려 추천이 오염되지 않는지 확인
```

**완료 기준:** guardrail 5문항 전부 통과, 인젝션 차단율 95% 이상

## 8단계: API

```
@PRD.md "API 스펙 (표준 규약)"을 그대로 따라 src/api.py 를 구현해줘.
- POST /query: 요청 {"question"} → 응답 {"answer", "contexts":[{"doc_id","text"}], "trace":[{"step","input","output"}]}
- 필드 이름과 구조는 규약에서 절대 바꾸지 말 것, 요청·응답은 Pydantic 모델
- contexts는 리랭킹 후 답변에 쓴 청크, 가드레일 차단 시 contexts는 빈 배열
- GET /health → {"status": "ok"}
httpx 기반 API 테스트를 작성하고, 정상·차단 질의 각 1건의 실제 응답 JSON을 보여줘.
```

**완료 기준:** 응답 구조가 규약과 일치, API 테스트 통과

## 9단계: Docker와 run.sh

```
@PRD.md Docker 설계 메모와 @README.md 실행 방법을 기준으로 Dockerfile, run.sh 를 작성해줘.
- 베이스 이미지 python:3.14.8-slim, WORKDIR /app
- 임베딩·리랭커 모델은 빌드 단계에서 미리 내려받기
- torch는 CPU 전용 빌드로 설치(--index-url https://download.pytorch.org/whl/cpu)해 GPU용 패키지가 들어가지 않게 할 것
- run.sh: 가격 DB 생성(data/prices.db 없을 때만, python -m src.seed_db) → 인덱싱(색인 없을 때만)
  → uvicorn(8000) → streamlit(8501)
- HEALTHCHECK는 GET /health 사용, .env는 이미지에 넣지 말고 --env-file로 주입
- 빌드 전에 새 가상환경에서 pip install --no-deps -r requirements.txt 로 클린 설치가 되는지 먼저 확인
  (Dockerfile도 --no-deps로 설치, Windows에서 고정한 목록이라 Linux에서만 필요한 패키지가 빠졌는지 확인)
README의 docker run 명령(볼륨 -v $(pwd)/data:/app/data 포함)으로 실제 실행해서
/health와 /query 응답을 확인하고, 이미지 크기와 빌드 시간을 보여줘.
README에 Windows PowerShell용 명령(-v "${PWD}/data:/app/data")도 함께 적어줘.
```

**완료 기준:** requirements.txt만으로 클린 설치 성공, 클린 환경에서 `docker run` 한 번으로 API 응답, README에 bash·PowerShell 실행 명령 모두 기재

## 10단계: UI

```
src/app.py 에 Streamlit 채팅 UI를 만들어줘.
- 입력한 질문을 POST /query로 보내고 answer를 표시
- contexts는 "근거 리뷰" 접기 영역에, trace는 "처리 과정" 접기 영역에 표시
- API 주소는 환경 변수로 받기
- 되묻기 대화 이어가기(@PRD.md): 응답이 되묻기이면 다음 입력을 받아 "이전 질문 + 되묻기 + 사용자 답"을
  하나의 question으로 이어 붙여 다시 보내기 (최대 2턴, API 규약은 그대로)
화면 구성은 단순하게, 데모 질의 3개를 버튼으로 넣어줘.
```

**완료 기준:** UI에서 추천 결과·근거·처리 과정 확인, 되묻기 후 답하면 추천까지 이어짐

## 11단계: 1차 평가 (Day 9)

```
@PRD.md "성공 지표"와 "test_queries.csv 스키마"를 기준으로 src/evaluate.py 를 구현하고 1차 평가를 실행해줘.
- REVIEW_SOURCE=sample, RETRIEVER_MODE=vector 로 30문항 전체를 POST /query로 호출
- 통과 판정: expected_traits 충족(LLM-as-Judge) + forbidden 미포함 + expected_tools 호출(trace 대조)
- 실행은 README 실행 방법과 같이 `python src/evaluate.py --queries evaluation/test_queries.csv --round 1`
- RAGAS 4지표: positive·edge 19문항 중 contexts가 있는(RAG를 거친) 문항만 대상,
  제외한 문항(예: 되묻기)과 이유를 리포트에 기록. reference는 expected_traits를 문장으로 이어 붙여 사용
- 30문항·RAGAS·Judge 호출에는 지수 백오프 재시도와 동시 호출 수 제한을 적용 (Bedrock ThrottlingException 대비)
  - RAGAS의 LLM·임베딩은 기본값(OpenAI)을 쓰지 말고 Bedrock LLM(JUDGE_MODEL_ID)과
    4단계 한국어 임베딩 모델을 래퍼로 명시 연결할 것
- LLM-as-Judge 5점 척도 채점, Judge 모델은 .env의 JUDGE_MODEL_ID (생성 모델 BEDROCK_MODEL_ID와 달라야 함)
- 문항별 응답 시간을 기록하고, 협찬 판별 정확도(sponsored_labels.csv)와
  인젝션 차단율(injection_tests.csv)도 같은 스크립트에서 다시 측정
- 결과를 evaluation/round1_report.md로 작성: PRD 성공 지표 8개(RAGAS 4, Judge, 협찬 정확도,
  차단율, 응답 시간)의 목표 대비 수치, 카테고리별 통과율, 실패 유형 분류
```

**완료 기준:** round1_report.md에 성공 지표 8개 기준선 수치와 실패 유형 정리 (이 시점 커밋 필수)

## 12단계: 개선과 2차 평가 (Day 10)

```
@evaluation/round1_report.md 의 실패 유형을 분석해서 개선 계획을 먼저 제안해줘.
기본 개선: RETRIEVER_MODE=hybrid(BM25+벡터, 리랭킹, 쿼리 확장), 가드레일 보완.
계획을 내가 승인하면 적용하고, 같은 test_queries.csv와 REVIEW_SOURCE=sample로 2차 평가를 실행해서
evaluation/round2_report.md를 작성해줘: 개선 조치별 지표 변화, 1차 대비 통과 건수 변화,
성공 지표 8개의 1차·2차 비교와 목표 달성 여부, 남은 한계.
```

**완료 기준:** 30문항 중 24문항 이상 + guardrail 5문항 전부 통과, 성공 지표 8개 목표 달성 여부 기록

## 13단계: 문서 마무리와 최종 검증

```
@README.md 의 TODO를 채워줘:
- 활용한 패턴의 Day 번호는 내가 알려줄게: (Day 번호 입력)
- RAGAS 결과와 인-아웃 통과율은 round2_report.md 수치 사용
- 핵심 코드 위치의 줄 번호(NN)를 실제 줄 번호로
- 트라이앤에러 회고는 git log와 round1/round2 리포트를 근거로 초안 작성
그다음 @PRD.md 기준으로 SERVICE.md, README.md가 실제 구현과 맞는지 검증하고 불일치를 표로 보여줘.
특히 쇼핑 검색 API를 사용한다는 서술이 남아 있지 않은지(종료 사실 설명은 제외), 실제/가짜 데이터 표기가 맞는지 확인해줘.
```

**완료 기준:** README TODO 0건, 세 문서와 구현 일치

---

## 진행 팁

- 단계마다 새 세션으로 시작하면 컨텍스트가 섞이지 않는다.
- 단계 완료마다 커밋하고, 11단계 직전 커밋은 1차 기준선으로 반드시 남긴다.
- 프롬프트 결과가 기대와 달라 수정한 내용은 바로 메모해 13단계 회고 재료로 쓴다.
