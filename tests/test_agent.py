"""agent.py 테스트 (LLM 호출 없음: 대본대로 도구를 부르는 가짜 LLM)."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import agent, chains, indexer
from src.chains import CardText, CardTexts, ConditionPoint, ReasonItem, ReviewAnalysis

QUESTION = "원룸에서 쓸 35만원 이하 무선청소기 추천해줘. 소음이 제일 중요해"


class ScriptedLLM:
    """bind_tools → 턴마다 대본의 AIMessage를 돌려준다. with_structured_output → 체인용 가짜 응답."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.agent_calls = 0
        self.structured: list[str] = []

    def bind_tools(self, _tools):
        def run(_messages):
            self.agent_calls += 1
            return self.turns.pop(0) if self.turns else AIMessage("끝")
        return RunnableLambda(run)

    def with_structured_output(self, schema):
        def run(prompt_value):
            self.structured.append(schema.__name__)
            text = prompt_value.to_string()
            if schema is ReviewAnalysis:
                mid = next(m for m in ("VC-M1", "VC-M3", "VC-M4") if f"분석 대상: {m}" in text)
                ids = [line.split('id="')[1].split('"')[0] for line in text.splitlines()
                       if line.startswith("<chunk")]
                return ReviewAnalysis(model_id=mid, verdicts=[], summary="조용함",
                                      pros=[ConditionPoint(condition="소음", point="조용", chunk_ids=ids[:1])])
            return CardTexts(cards=[CardText(model_id="VC-M1", summary="요약",
                                             reasons=[ReasonItem(text="조용")])])
        return RunnableLambda(run)


def _call(name, args, i):
    return {"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}


@pytest.fixture
def llm(monkeypatch):
    def install(turns):
        fake = ScriptedLLM(turns)
        chains.set_llm(fake)
        return fake
    yield install
    chains.set_llm(None)


@pytest.fixture
def built():
    return indexer.build_index()


def test_clarify_without_tools(llm):
    fake = llm([AIMessage("예산과 사용 환경(원룸·반려동물 등)을 알려주세요.")])
    r = agent.run("청소기 추천해줘")
    assert r.answer.startswith("예산과 사용 환경")
    assert r.trace == [{"step": "answer", "input": "에이전트 응답", "output": "되묻기"}]
    assert r.contexts == [] and fake.agent_calls == 1


def test_recommend_flow_fills_args_and_skips_final_llm_turn(llm, built):
    fake = llm([
        AIMessage("", tool_calls=[_call("search_products", {"keyword": "무선청소기", "max_price": 350000}, 1)]),
        AIMessage("", tool_calls=[
            _call("fetch_reviews", {"product_names": ["클린웨이브 V7 사일런트 무선청소기"], "conditions": ["소음"]}, 2),
            _call("retrieve_reviews", {"queries": [QUESTION, "소음"]}, 3),          # model_ids 생략
            _call("calc_real_price", {"user_cards": ["한결카드"]}, 4),             # items 생략
        ]),
    ])
    r = agent.run(QUESTION)
    steps = [s["step"] for s in r.trace]
    assert steps == ["search_products", "fetch_reviews", "retrieve_reviews", "calc_real_price", "answer"]
    assert fake.agent_calls == 2                                  # 결과가 모이면 LLM 턴 없이 answer
    assert "model_ids ['VC-M" in r.trace[2]["input"]             # 후보 모델로 보강
    assert r.trace[3]["input"].startswith("판매처별 상품 ") and "0건" not in r.trace[3]["input"]
    assert "샘플 데이터" in r.answer and "판매처별 실결제가" in r.answer
    assert r.contexts and all(c["doc_id"].startswith("rv-") for c in r.contexts)
    assert fake.structured.count("ReviewAnalysis") >= 1 and "CardTexts" in fake.structured


def test_tool_limit_and_search_limit(llm, built):
    turns = [AIMessage("", tool_calls=[_call("search_products", {"keyword": f"무선청소기{i}"}, i)])
             for i in range(5)]
    turns.append(AIMessage("", tool_calls=[_call("query_price_db", {"product_ids": ["VC-001"]}, j)
                                           for j in range(10, 15)]))
    llm(turns)
    r = agent.run("무선청소기 가격 추이")
    tool_steps = [s for s in r.trace if s["step"] != "answer"]
    assert sum(s["step"] == "search_products" for s in tool_steps) == agent.MAX_SEARCHES
    assert len(tool_steps) <= agent.MAX_TOOL_CALLS


def test_price_only_path(llm, built):
    fake = llm([
        AIMessage("", tool_calls=[_call("search_products", {"keyword": "클린웨이브 V7 사일런트"}, 1)]),
        AIMessage("", tool_calls=[_call("calc_real_price", {"user_memberships": ["바로플러스"]}, 2)]),
        AIMessage("바로마켓이 가장 쌉니다."),
    ])
    r = agent.run("클린웨이브 V7 사일런트 어디서 사는 게 제일 싸? 바로플러스 있어")
    assert r.answer.startswith("판매처별 실결제가 비교")
    assert r.contexts == [] and r.trace[-1]["output"].startswith("실결제가 비교")
    assert fake.agent_calls == 2                     # 마지막 단계 도구 뒤에는 LLM 마무리 턴 없음
    assert "전기포트" not in r.answer and "클린웨이브 V7 사일런트" in r.answer


def test_render_price_history_verdicts():
    h = [{"product_id": "P1", "model_id": "M", "mall": "몰", "current_price": 100000, "min_price": 99000,
          "min_date": "2026-11-01", "avg_price": 105000, "max_price": 110000},
         {"product_id": "P2", "model_id": "M", "mall": "몰2", "current_price": 120000, "min_price": 100000,
          "min_date": "2026-11-01", "avg_price": 110000, "max_price": 125000}]
    text = agent.render_price_history(h, "2026-10-13 ~ 2026-11-11", {})
    assert "최저가 수준" in text and "할인을 기다려 볼 만합니다" in text and "샘플 데이터" in text
