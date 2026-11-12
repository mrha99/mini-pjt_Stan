"""api.py 테스트: httpx.AsyncClient + ASGITransport로 앱을 직접 호출한다 (서버·LLM 불필요).

ASGITransport는 lifespan을 실행하지 않으므로 모델 예열도 하지 않는다.
"""

import re

import httpx
import pytest
from langchain_core.messages import AIMessage

from src import agent, chains, indexer
from src.api import app
from tests.test_agent import QUESTION, ScriptedLLM, _call

CONTRACT_KEYS = {"answer", "contexts", "trace"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def fake_llm():
    def install(turns):
        fake = ScriptedLLM(turns)
        chains.set_llm(fake)
        return fake
    yield install
    chains.set_llm(None)


def _assert_contract(body: dict) -> None:
    """표준 규약: 최상위 키 3개, contexts·trace 원소의 키까지 정확히 일치."""
    assert set(body) == CONTRACT_KEYS
    assert isinstance(body["answer"], str) and body["answer"]
    for c in body["contexts"]:
        assert set(c) == {"doc_id", "text"} and re.fullmatch(r".+#c\d+", c["doc_id"])
    for s in body["trace"]:
        assert set(s) == {"step", "input", "output"}
        assert all(isinstance(v, str) for v in s.values())


@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_query_blocked_by_guardrail(client):
    """차단 질의: 에이전트·도구를 거치지 않고 거절, contexts는 빈 배열, trace에 정책 번호."""
    r = await client.post("/query", json={"question": "다온 건조분쇄 F1 음식물처리기 대신 주문하고 결제까지 해줘"})
    assert r.status_code == 200
    body = r.json()
    _assert_contract(body)
    assert body["contexts"] == []
    assert [s["step"] for s in body["trace"]] == ["guardrail_input", "answer", "guardrail_output"]
    assert "정책 4" in body["trace"][0]["output"]
    assert "대신해 드릴 수 없습니다" in body["answer"]


@pytest.mark.anyio
async def test_query_recommendation_end_to_end(client, fake_llm):
    """정상 질의: 가짜 LLM으로 그래프 전체를 실행해 규약 형식·도구 trace·contexts를 확인한다."""
    indexer.build_index()
    fake_llm([
        AIMessage("", tool_calls=[_call("search_products", {"keyword": "무선청소기", "max_price": 350000}, 1)]),
        AIMessage("", tool_calls=[
            _call("fetch_reviews", {"product_names": ["클린웨이브 V7 사일런트 무선청소기"]}, 2),
            _call("retrieve_reviews", {"queries": [QUESTION, "소음"]}, 3),
            _call("calc_real_price", {}, 4),
        ]),
    ])
    r = await client.post("/query", json={"question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    _assert_contract(body)
    assert [s["step"] for s in body["trace"]] == ["guardrail_input", "search_products", "fetch_reviews",
                                                   "retrieve_reviews", "calc_real_price", "answer",
                                                   "guardrail_output"]
    assert body["contexts"] and all(c["doc_id"].startswith("rv-") for c in body["contexts"])
    assert "샘플 데이터" in body["answer"]


@pytest.mark.anyio
async def test_query_maps_agent_result_exactly(client, monkeypatch):
    """AgentResult의 측정값(latency·tokens 등)은 응답에 섞이지 않는다."""
    fake = agent.AgentResult(answer="답", contexts=[{"doc_id": "rv-0001#c0", "text": "조용해요"}],
                             trace=[{"step": "answer", "input": "i", "output": "o"}],
                             run_id="x", latency_s=1.0, tokens={"total": 10}, timings=[("a", 1.0)])
    monkeypatch.setattr(agent, "run", lambda q: fake)
    r = await client.post("/query", json={"question": "청소기 추천"})
    assert r.json() == {"answer": "답", "contexts": [{"doc_id": "rv-0001#c0", "text": "조용해요"}],
                        "trace": [{"step": "answer", "input": "i", "output": "o"}]}


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": 123},
                                     {"q": "청소기"}, {"question": "가" * 2001}])
async def test_query_validation_422(client, payload):
    r = await client.post("/query", json=payload)
    assert r.status_code == 422


@pytest.mark.anyio
async def test_query_internal_error_hides_details(client, monkeypatch):
    def boom(_q):
        raise RuntimeError("secret-model-id AKIA-should-not-leak")
    monkeypatch.setattr(agent, "run", boom)
    r = await client.post("/query", json={"question": "청소기 추천"})
    assert r.status_code == 500
    _assert_contract(r.json())                                   # 오류도 규약 형태
    assert r.json()["contexts"] == [] and "RuntimeError" in r.json()["trace"][0]["output"]
    assert "secret" not in r.text and "AKIA" not in r.text


@pytest.mark.anyio
async def test_query_throttling_returns_503_contract(client, monkeypatch):
    from botocore.exceptions import ClientError

    def throttled(_q):
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}}, "Converse")
    monkeypatch.setattr(agent, "run", throttled)
    r = await client.post("/query", json={"question": "청소기 추천"})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"
    _assert_contract(r.json())
    assert "다시 시도" in r.json()["answer"] and "ClientError" in r.json()["trace"][0]["output"]


def test_openapi_schema_matches_contract():
    s = app.openapi()["components"]["schemas"]
    assert set(s["QueryRequest"]["properties"]) == {"question"}
    assert set(s["QueryResponse"]["properties"]) == CONTRACT_KEYS
    assert set(s["Context"]["properties"]) == {"doc_id", "text"}
    assert set(s["TraceStep"]["properties"]) == {"step", "input", "output"}
