"""FastAPI POST /query (표준 규약), GET /health

PRD "API 스펙 (표준 규약)"을 그대로 따른다. 필드 이름·구조는 바꾸지 않는다.

    POST /query   {"question": "..."}
               → {"answer": "...",
                  "contexts": [{"doc_id": "{리뷰ID}#c{번호}", "text": "..."}],
                  "trace": [{"step": "...", "input": "...", "output": "..."}]}
    GET  /health  → {"status": "ok"}   (Docker 헬스체크용 보조 엔드포인트)

- contexts: 리랭킹 후 답변에 쓴 리뷰 청크 (정화본). 가드레일 차단·되묻기·가격만 묻는 질문은 빈 배열.
- trace[].step: guardrail_input, 도구 이름 5종, answer, guardrail_output. 차단 시 SERVICE.md 정책 번호 포함.
- API는 대화 상태를 저장하지 않는다. 질문 원문은 로그에 남기지 않는다 (SERVICE.md 정책 3).
- 서버 시작 시 임베딩·리랭커를 예열한다(API_WARMUP=0이면 생략). 예열이 끝나야 /health가 응답한다.
- 오류도 규약 형태로 돌려준다(평가·UI가 같은 파서를 쓰게): Bedrock 호출 한도·일시 오류는 503 + Retry-After,
  그 밖은 500. answer에 안내, contexts는 빈 배열, trace에 오류 유형(내부 메시지·키·모델 ID는 싣지 않음).

실행: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src import agent, chains

logger = logging.getLogger("shopping_agent.api")

MAX_QUESTION_CHARS = 2000


# ─────────────────────────────── 요청·응답 모델 (표준 규약) ───────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS, description="사용자 질의")


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="리뷰 청크 ID ({리뷰ID}#c{번호})")
    text: str = Field(description="리랭킹 후 답변에 쓴 리뷰 청크 (지시문 제거·PII 마스킹)")


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str = Field(description="guardrail_input · 도구 이름 · answer · guardrail_output")
    input: str
    output: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    contexts: list[Context]
    trace: list[TraceStep]


class HealthResponse(BaseModel):
    status: str = "ok"


# ─────────────────────────────── 앱 ───────────────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("API_WARMUP", "1") != "0":
        secs = agent.warmup()
        logger.info("모델 예열 완료 %.1f초", secs)
    yield


app = FastAPI(title="진짜후기 쇼핑 비교 에이전트", version="1.0.0", lifespan=lifespan,
              description="협찬 리뷰를 걸러낸 실구매 후기와 실결제가로 Top 3를 추천한다 (가격·판매처는 샘플 데이터).")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _error_response(exc: Exception) -> JSONResponse:
    transient = chains.is_retryable(exc)
    answer = ("요청이 많아 지금은 답변을 만들지 못했습니다. 30초쯤 뒤에 다시 시도해 주세요." if transient
              else "일시적인 오류로 답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.")
    body = QueryResponse(answer=answer, contexts=[],
                         trace=[TraceStep(step="answer", input="에이전트 실행", output=f"오류: {type(exc).__name__}")])
    return JSONResponse(status_code=503 if transient else 500, content=body.model_dump(),
                        headers={"Retry-After": "30"} if transient else None)


@app.post("/query", response_model=QueryResponse,
          responses={503: {"model": QueryResponse, "description": "LLM 호출 한도 등 일시 오류"},
                     500: {"model": QueryResponse, "description": "내부 오류"}})
def query(req: QueryRequest) -> QueryResponse | JSONResponse:
    """질문 1건 처리 (무상태). 동기 함수라 FastAPI 스레드풀에서 실행된다."""
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question이 비어 있습니다.")
    t = time.perf_counter()
    try:
        result = agent.run(req.question)
    except Exception as exc:  # noqa: BLE001 - 내부 오류 내용(키·모델 ID 등)은 응답에 싣지 않는다
        logger.exception("query 실패: %s", type(exc).__name__)
        return _error_response(exc)
    logger.info("query %.1f초 · steps=%s", time.perf_counter() - t, [s["step"] for s in result.trace])
    return QueryResponse(
        answer=result.answer,
        contexts=[Context(doc_id=c["doc_id"], text=c["text"]) for c in result.contexts],
        trace=[TraceStep(step=s["step"], input=s["input"], output=s["output"]) for s in result.trace],
    )
