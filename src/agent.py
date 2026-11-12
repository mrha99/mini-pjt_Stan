"""ReAct 에이전트 그래프 (패턴 2) + LangSmith trace (패턴 11)

LangGraph StateGraph로 만든 단일 ReAct 에이전트다.

    START → guardrail_input ─(차단)─────────────────────────────────────────────┐
                    └→ agent(LLM: 도구 선택) ⇄ tools(도구 실행) → answer → guardrail_output → END

- guardrail_input / guardrail_output: SERVICE.md 정책 5건 (src/guardrails.py). 차단되면 에이전트를 거치지 않고
  거절 + 할 수 있는 일을 안내하며 trace에 정책 번호를 남긴다. 질문의 PII는 그래프에 넣기 전에 마스킹한다
  (LangSmith에도 남지 않게, 정책 3). 리뷰 텍스트는 retrieve_reviews 결과와 chains._wrap에서 정화된다(정책 2).

- agent: Claude(ChatBedrockConverse)에 도구 5종을 바인딩한다. 조건이 부족하면 도구 없이 되묻는다.
- tools: LLM이 요청한 도구를 요청 순서대로 실행한다 (fetch_reviews로 색인 → retrieve_reviews로 검색).
  도구 원본 결과는 그래프 상태에 두고 LLM에는 요약만 넘긴다. 가격·할인 정보는 LLM이 옮겨 적지 않도록
  calc_real_price의 items 생략 시 search_products 후보 전체로, retrieve_reviews의 model_ids 생략 시
  후보 모델로 채운다. 요청당 도구 호출 최대 6회, search_products 최대 3회(재검색 2회).
- answer: 리뷰 검색과 실결제가가 모이면 LLM 턴 없이 바로 온다. 상태에 모인 결과로 경로를 고른다
  (추천 카드 / 실결제가 비교 / 실구매 후기 요약 / 가격 이력, 해당 없으면 에이전트의 마지막 답변).

에이전트는 대화 상태를 저장하지 않는다. "이전 질문 + 되묻기 + 사용자 답"이 이어 붙은 question은
하나의 요청으로 처리한다 (PRD "되묻기 대화 이어가기").

CLI
    python -m src.agent "원룸에서 쓸 35만원 이하 무선청소기 추천해줘"
    python -m src.agent --ids 1,8,10          # evaluation/test_queries.csv 문항 실행
"""

from __future__ import annotations

import argparse
import csv
import json
import operator
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src import chains, guardrails, retriever, tools
from src.guardrails import mask_pii

ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = ROOT / "evaluation" / "test_queries.csv"

MAX_TOOL_CALLS = 6
MAX_SEARCHES = 3          # 첫 검색 + 재검색 2회
MIN_MODELS = 3            # Top 3
MAX_LLM_TURNS = 6         # 무한 루프 방지 (도구 한도와 별개)
RESPONSE_BUDGET_S = 30.0


# ─────────────────────────────── 상태 ───────────────────────────────


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    products: dict[str, dict]        # product_id → search_products 상품 (누적)
    model_ids: list[str]             # 후보 모델 (검색 순, 누적)
    review_model_ids: list[str]      # fetch_reviews가 찾은 모델
    retrieved: bool                  # retrieve_reviews 실행 여부
    chunks: list[dict]               # 답변 근거 청크 (sample)
    blog_refs: list[dict]            # 실제 블로그 참고 후기 (근거 아님)
    real_prices: dict[str, dict]     # product_id → calc_real_price 결과
    price_histories: list[dict]
    price_period: str
    notices: list[str]
    tool_calls: int
    searches: int
    llm_turns: int
    trace: Annotated[list[dict], operator.add]
    timings: Annotated[list[tuple[str, float]], operator.add]
    contexts: list[dict]
    answer: str
    blocked: bool                    # 입력 가드레일 차단 여부
    answer_source: str               # code(도구 결과 렌더링) / llm(에이전트 서술) / guardrail(거절 안내)


class AgentResult(BaseModel):
    """API 응답(answer·contexts·trace) + CLI용 측정값."""

    answer: str
    contexts: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    run_id: str = ""
    latency_s: float = 0.0
    tokens: dict = Field(default_factory=dict, description="로컬 집계 토큰 (input/output/total)")
    langsmith: dict = Field(default_factory=dict, description="LangSmith 조회값 (total_tokens·latency_s)")
    timings: list[tuple[str, float]] = Field(default_factory=list)


# ─────────────────────────────── LLM·도구 바인딩 ───────────────────────────────


class CalcRealPriceAgentInput(BaseModel):
    """에이전트용 calc_real_price 입력: items를 생략하면 search_products 후보 전체로 채운다."""

    items: list[tools.PriceItem] | None = Field(
        default=None, description="생략 권장. 비우면 search_products로 찾은 후보 상품 전체를 계산한다")
    user_cards: list[str] = Field(default_factory=list, description="사용자가 가진 카드 (없으면 빈 목록)")
    user_memberships: list[str] = Field(default_factory=list, description="사용자가 가진 멤버십")


def _calc_schema_only(**_: Any) -> dict:  # 바인딩용 (실행은 tools 노드가 tools.calc_real_price로)
    raise RuntimeError("tools 노드에서 실행한다")


CALC_AGENT_TOOL = StructuredTool.from_function(
    func=_calc_schema_only, name="calc_real_price", args_schema=CalcRealPriceAgentInput,
    description=("판매처별 실결제가(판매가 - 멤버십·카드 할인 + 배송비)를 계산한다. items는 비워 두면 "
                 "search_products 후보 전체로 채워진다. 사용자가 말한 카드·멤버십만 넣는다."))

AGENT_TOOLS = [tools.search_products, tools.fetch_reviews, tools.retrieve_reviews,
               tools.query_price_db, CALC_AGENT_TOOL]
TOOL_IMPL = {t.name: t for t in tools.ALL_TOOLS}


def _categories() -> list[str]:
    from src import indexer

    return sorted({p["category"] for p in indexer.catalog().values()})


SYSTEM_PROMPT = """너는 국내 쇼핑 비교 에이전트다. 도구로 상품·리뷰·가격을 모으면, 최종 답변(추천 카드·가격 비교·후기 요약)은
시스템이 도구 결과로 만든다. 상품·가격·판매처는 샘플 데이터다. 카탈로그 카테고리: {categories}

질문에 "이전 질문 / 되묻기 / 사용자 답"이 이어 붙어 있으면 전체를 하나의 요청으로 본다.

1. 질문에서 카테고리(또는 상품명), 예산(최대 가격), 용도·조건, 보유 카드·멤버십을 파악한다.
2. 상품 추천 요청인데 예산이나 용도(사용 환경)가 빠졌거나 "적당한 가격"처럼 모호하면 도구를 부르지 말고
   필요한 것만 1~2개 짧게 되묻는다. 특정 상품의 가격·후기·스펙 질문은 되묻지 않는다.
3. 첫 턴: search_products(keyword=카테고리명 또는 상품명, max_price=예산). 카탈로그에 없는 카테고리는
   비슷한 다른 카테고리로 바꿔 검색하지 않는다. 상품 두 개를 비교하면 상품마다 한 번씩 검색한다.
4. 추천 요청에서 모델이 3개 미만이면 검색어를 바꿔(브랜드·수식어 빼기 등) 재검색할 수 있다(최대 2회).
   예산 상한은 올리지 않는다. 그래도 부족하면 찾은 모델로 진행한다.
5. 다음 턴에 필요한 도구를 한 번에 호출한다 (여러 도구를 같은 턴에 요청하면 요청 순서대로 실행된다):
   - 추천·비교: fetch_reviews(후보 상품명 전체, conditions=조건어) → retrieve_reviews(queries=[사용자 질문, 조건어...],
     model_ids=후보 model_id) → calc_real_price(items 비움, user_cards, user_memberships)
   - 어디서 사는 게 싼지(실결제가)만: calc_real_price
   - 가격 추이·지금 살 때인지: query_price_db(후보 product_ids)
   - 특정 상품 후기·장단점: fetch_reviews(상품명) → retrieve_reviews(queries=[질문, 조건어])
6. 가격·스펙·수치는 도구 결과에 있는 것만 쓴다. 도구 결과에 없는 스펙은 "확인 불가"라고 답한다.
   상품을 못 찾으면 못 찾았다고 하고 다른 검색어를 제안한다.
7. 필요한 도구를 다 불렀으면 도구 없이 2~3문장으로 마무리한다 (가격·순위는 시스템이 붙인다)."""


def _agent_runnable():
    llm = chains.get_llm(max_tokens=1000)
    return chains._guarded(llm.bind_tools(AGENT_TOOLS), "agent_llm")


# ─────────────────────────────── 요약 헬퍼 (LLM·trace용) ───────────────────────────────


def _won(n: int) -> str:
    return f"{n:,}원"


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return mask_pii(text if len(text) <= n else text[: n - 1] + "…")


def _args_summary(name: str, args: dict) -> str:
    if name == "search_products":
        rng = "~".join(_won(v) if v else "" for v in (args.get("min_price"), args.get("max_price")))
        return f"{args.get('keyword')}" + (f", {rng}" if args.get("min_price") or args.get("max_price") else "")
    if name == "fetch_reviews":
        names = args.get("product_names", [])
        return f"상품명 {len(names)}개: {', '.join(names)}" + (
            f" / 조건어 {', '.join(args['conditions'])}" if args.get("conditions") else "")
    if name == "retrieve_reviews":
        return f"쿼리 {len(args.get('queries', []))}개: {' | '.join(args.get('queries', []))} / model_ids {args.get('model_ids')}"
    if name == "calc_real_price":
        return (f"판매처별 상품 {len(args.get('items') or [])}건, 카드 {args.get('user_cards') or '없음'}, "
                f"멤버십 {args.get('user_memberships') or '없음'}")
    if name == "query_price_db":
        return f"상품 {', '.join(args.get('product_ids', []))}, {args.get('days', 30)}일"
    return json.dumps(args, ensure_ascii=False)


def _model_lines(products: list[dict]) -> list[str]:
    by_model: dict[str, list[dict]] = {}
    for p in products:
        by_model.setdefault(p["model_id"], []).append(p)
    lines = []
    for mid, ps in by_model.items():
        prices = [p["price"] for p in ps]
        lines.append(f"- {mid} {ps[0]['name']} | 판매처 {len(ps)}곳 | 판매가 {_won(min(prices))}~{_won(max(prices))}"
                     f" | product_ids {','.join(p['product_id'] for p in ps)}"
                     f" | 스펙 {json.dumps(ps[0].get('specs', {}), ensure_ascii=False)}")
    return lines


# ─────────────────────────────── 노드: agent ───────────────────────────────


def agent_node(state: AgentState) -> dict:
    t = time.perf_counter()
    system = SystemMessage(SYSTEM_PROMPT.format(categories=", ".join(_categories())))
    msg: AIMessage = _agent_runnable().invoke([system, *state["messages"]])
    turn = state.get("llm_turns", 0) + 1
    return {"messages": [msg], "llm_turns": turn, "timings": [(f"agent_llm#{turn}", time.perf_counter() - t)]}


# ─────────────────────────────── 노드: tools ───────────────────────────────


def _run_tool(name: str, args: dict, st: dict) -> tuple[str, str, dict]:
    """도구 1건 실행 → (LLM용 요약, trace output, 상태 갱신분). st는 이번 라운드의 누적 상태."""
    if name == "calc_real_price" and not args.get("items"):
        args["items"] = [{k: p[k] for k in ("product_id", "model_id", "mall", "price", "shipping_fee",
                                           "card_discounts", "membership_discounts")}
                         for p in st["products"].values()]
        if not args["items"]:
            return ("계산할 상품이 없습니다. 먼저 search_products로 상품을 찾으세요.", "상품 없음", {})
    if name == "retrieve_reviews" and not args.get("model_ids"):
        ids = st["model_ids"] or st["review_model_ids"]
        if ids:
            args["model_ids"] = list(ids)

    out = TOOL_IMPL[name].invoke(args)
    upd: dict = {}
    if name == "search_products":
        new = {p["product_id"]: p for p in out["products"]}
        products = {**st["products"], **new}
        model_ids = list(dict.fromkeys(st["model_ids"] + out["model_ids"]))
        upd = {"products": products, "model_ids": model_ids, "searches": st["searches"] + 1}
        left = MAX_SEARCHES - upd["searches"]
        hint = ""
        if len(model_ids) < MIN_MODELS:
            hint = (f"\n안내: 누적 후보 모델이 {len(model_ids)}개로 3개 미만입니다. "
                    + (f"추천 요청이면 검색어를 바꿔 재검색할 수 있습니다(남은 재검색 {left}회, 예산 상한 유지)."
                       if left > 0 else "재검색 한도를 모두 썼으니 찾은 모델로 진행하세요."))
        content = (f"상품 {out['count']}개 · 모델 {len(out['model_ids'])}개 (누적 후보 모델 {len(model_ids)}개)\n"
                   + "\n".join(_model_lines(out["products"])) + hint) if out["count"] else (
            "검색 결과가 없습니다 (해당 상품·카테고리가 샘플 카탈로그에 없음)." + hint)
        return content, f"후보 {out['count']}개 (모델 {len(out['model_ids'])}개)", upd
    if name == "fetch_reviews":
        upd = {"review_model_ids": list(dict.fromkeys(st["review_model_ids"] + out.get("model_ids", [])))}
        if out.get("notice") and out.get("from_cache"):
            upd["notices"] = st["notices"] + [out["notice"]]
        content = (f"리뷰 {out['count']}건 수집·색인 (모델 {out.get('model_ids')}) · 리뷰 없는 모델 "
                   f"{out.get('no_review_models') or '없음'} · 못 찾은 상품명 {out.get('not_found') or '없음'}"
                   f" · {out.get('notice', '')}")
        return content, f"리뷰 {out['count']}건 색인 (리뷰 없는 모델 {len(out.get('no_review_models', []))}개)", upd
    if name == "retrieve_reviews":
        chunks = [c for c in out["chunks"] if c.get("source") != "blog"]
        upd = {"retrieved": True, "chunks": chunks, "blog_refs": out.get("blog_references", [])}
        per = {}
        for c in chunks:
            per[c.get("model_id")] = per.get(c.get("model_id"), 0) + 1
        content = (f"리뷰 청크 {len(chunks)}개 (모델별 {per or '없음'}) · 상태 {out.get('status')} · "
                   f"{out.get('notice', '')}")
        return content, f"리뷰 청크 {len(chunks)}개 (리랭킹 후, 모델 {len(per)}개)", upd
    if name == "calc_real_price":
        real = {**st["real_prices"], **{r["product_id"]: r for r in out["results"]}}
        upd = {"real_prices": real}
        lines = [f"- {r['product_id']} {r.get('mall')}: 실결제가 {_won(r['real_price'])}"
                 f" (할인 {', '.join(a['name'] for a in r['applied']) or '없음'})" for r in out["results"]]
        return "실결제가 계산 완료\n" + "\n".join(lines), f"실결제가 {len(out['results'])}건", upd
    if name == "query_price_db":
        upd = {"price_histories": out["histories"], "price_period": out.get("period", "")}
        lines = [f"- {h['product_id']} {h['mall']}: 현재 {_won(h['current_price'])}, 최저 {_won(h['min_price'])}"
                 f"({h['min_date']}), 평균 {_won(h['avg_price'])}, 최고 {_won(h['max_price'])}"
                 for h in out["histories"]]
        return (f"가격 이력 ({out.get('period')})\n" + "\n".join(lines)
                + (f"\n없는 상품: {out['not_found']}" if out.get("not_found") else "")), \
            f"가격 이력 {len(out['histories'])}건", upd
    return json.dumps(out, ensure_ascii=False)[:2000], "완료", upd


def tools_node(state: AgentState) -> dict:
    last: AIMessage = state["messages"][-1]
    st = {
        "products": dict(state.get("products") or {}), "model_ids": list(state.get("model_ids") or []),
        "review_model_ids": list(state.get("review_model_ids") or []),
        "real_prices": dict(state.get("real_prices") or {}), "notices": list(state.get("notices") or []),
        "searches": state.get("searches", 0), "tool_calls": state.get("tool_calls", 0),
    }
    extra: dict = {}
    messages, trace, timings = [], [], []
    for call in last.tool_calls:
        name, args = call["name"], dict(call.get("args") or {})
        if st["tool_calls"] >= MAX_TOOL_CALLS:
            content, output = f"도구 호출 한도({MAX_TOOL_CALLS}회)에 도달해 실행하지 않았습니다.", "한도 초과로 생략"
        elif name == "search_products" and st["searches"] >= MAX_SEARCHES:
            content, output = "재검색 한도(2회)를 모두 썼습니다. 찾은 후보로 진행하세요.", "재검색 한도로 생략"
        elif name not in TOOL_IMPL:
            content, output = f"알 수 없는 도구: {name}", "알 수 없는 도구"
        else:
            st["tool_calls"] += 1
            t = time.perf_counter()
            try:
                content, output, upd = _run_tool(name, args, st)
            except Exception as exc:  # noqa: BLE001 - 에이전트가 오류를 보고 다음 행동을 고르게
                content, output, upd = f"도구 오류: {type(exc).__name__}", f"오류 {type(exc).__name__}", {}
            timings.append((f"tool:{name}", time.perf_counter() - t))
            for k, v in upd.items():
                if k in st:
                    st[k] = v
                else:
                    extra[k] = v
            trace.append({"step": name, "input": _short(_args_summary(name, args)), "output": _short(output)})
        messages.append(ToolMessage(content=mask_pii(content), tool_call_id=call["id"], name=name))
    return {"messages": messages, "trace": trace, "timings": timings, **st, **extra}


# ─────────────────────────────── 노드: answer ───────────────────────────────


def render_price_history(histories: list[dict], period: str, products: dict[str, dict]) -> str:
    """30일 가격 추이와 현재가 판단 (코드 계산, 샘플 데이터)."""
    if not histories:
        return ""
    lines = [f"최근 가격 흐름 ({period}, 샘플 데이터·참고용):"]
    for h in sorted(histories, key=lambda x: x["current_price"]):
        cur, low, avg = h["current_price"], h["min_price"], h["avg_price"]
        if cur <= low * 1.02:
            verdict = "30일 최저가 수준이라 지금 사도 괜찮은 편입니다"
        elif cur <= avg:
            verdict = "30일 평균보다 낮아 나쁘지 않은 가격입니다"
        else:
            verdict = f"30일 평균보다 {_won(cur - avg)} 높아 할인을 기다려 볼 만합니다"
        name = products.get(h["product_id"], {}).get("name", h["model_id"])
        lines.append(f"- {name} · {h['mall']}({h['product_id']}): 현재 {_won(cur)} · 최저 {_won(low)}"
                     f"({h['min_date']}) · 평균 {_won(avg)} · 최고 {_won(h['max_price'])} → {verdict}")
    return "\n".join(lines)


def _final_ai_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return (m.text if isinstance(m.text, str) else str(m.content)).strip()
    return ""


def answer_node(state: AgentState) -> dict:
    q = state["question"]
    conditions = retriever.extract_conditions([q])
    products = state.get("products") or {}
    model_ids = state.get("model_ids") or []
    chunks = state.get("chunks") or []
    real = list((state.get("real_prices") or {}).values())
    notices = list(state.get("notices") or [])
    category = next(iter(products.values()), {}).get("category") or retriever._category_of(q) or ""
    timings: list[tuple[str, float]] = []
    parts: list[str] = []
    contexts: list[dict] = []

    def timed(label, fn, *a, **k):
        t = time.perf_counter()
        out = fn(*a, **k)
        timings.append((label, time.perf_counter() - t))
        return out

    if state.get("retrieved") and real:
        ids = [m for m in model_ids if any(p["model_id"] == m for p in products.values())]
        analyses = timed("answer:review_analysis", chains.analyze_models, q, chunks, ids, conditions)
        cards = timed("answer:cards", chains.build_cards, q, list(products.values()), real,
                      analyses, chunks, conditions)
        parts.append(chains.render_answer(cards, category=category, notices=notices,
                                          blog_refs=state.get("blog_refs")))
        contexts = chunks
        summary = (f"후보 모델 {len(ids)}개 + 리뷰 청크 {len(chunks)}개", f"추천 카드 {len(cards)}개")
    elif real:
        cards = chains.build_cards(q, list(products.values()), real, {}, [], conditions,
                                   top_n=len(model_ids) or 10)
        parts.append(chains.render_price_answer(cards, notices))
        summary = (f"판매처 {len(real)}곳 실결제가", f"실결제가 비교 {len(cards)}개 모델")
    elif state.get("retrieved"):
        ids = state.get("review_model_ids") or model_ids or list(dict.fromkeys(c["model_id"] for c in chunks))
        analyses = timed("answer:review_analysis", chains.analyze_models, q, chunks, ids, conditions)
        parts.append(chains.render_review_summary(analyses, chunks, notices))
        contexts = chunks
        summary = (f"모델 {len(ids)}개 + 리뷰 청크 {len(chunks)}개", f"실구매 후기 요약 {len(analyses)}개 모델")
    else:
        summary = None

    if state.get("price_histories"):
        parts.append(render_price_history(state["price_histories"], state.get("price_period", ""), products))
        summary = summary or (f"가격 이력 {len(state['price_histories'])}건", "가격 추이 비교")

    tool_calls = state.get("tool_calls", 0)
    if parts:
        answer, source = "\n\n".join(parts), "code"
    else:
        answer = _final_ai_text(state) or "조건에 맞는 상품을 찾지 못했습니다. 다른 상품명이나 조건으로 다시 질문해 주세요."
        source = "llm"
        summary = ("에이전트 응답", "되묻기" if tool_calls == 0 else "안내 (도구 결과 기반)")
    missing = [n for n in notices if n not in answer]   # 가드레일 안내(정책 1 등)는 어느 경로든 붙인다
    if missing:
        answer += "\n" + "\n".join(f"※ {n}" for n in missing)
    step = {"step": "answer", "input": _short(summary[0]), "output": _short(summary[1])}
    return {"answer": answer, "answer_source": source,
            "contexts": [{"doc_id": c["doc_id"], "text": c["text"]} for c in contexts],
            "trace": [step], "timings": timings}


# ─────────────────────────────── 노드: 가드레일 ───────────────────────────────


def guardrail_input_node(state: AgentState) -> dict:
    t = time.perf_counter()
    v = guardrails.check_input(state["question"])
    trace = [{"step": "guardrail_input", "input": _short(state["question"]), "output": _short(v.trace_output(), 200)}]
    timing = [("guardrail_input", time.perf_counter() - t)]
    if not v.allowed:
        trace.append({"step": "answer", "input": "가드레일 차단",
                      "output": f"거절 + 할 수 있는 일 안내 (SERVICE.md 정책 {v.policy})"})
        return {"blocked": True, "answer": v.answer, "answer_source": "guardrail", "contexts": [],
                "trace": trace, "timings": timing}
    return {"blocked": False, "notices": list(v.notices), "trace": trace, "timings": timing}


def _allowed_amounts(state: AgentState) -> set[int]:
    """도구 결과·질문에 나온 금액 (LLM 서술 답변의 금액 검증용, 정책 1)."""
    amounts: set[int] = set(guardrails.amounts_in(state.get("question", "")))
    for p in (state.get("products") or {}).values():
        amounts |= {p["price"], p.get("shipping_fee", 0)}
        for d in p.get("card_discounts", []) + p.get("membership_discounts", []):
            amounts |= {v for v in (d.get("max_discount"), d.get("min_amount")) if v}
            if d.get("type") == "amount":
                amounts.add(int(d["value"]))
    for r in (state.get("real_prices") or {}).values():
        amounts |= {r["price"], r["shipping_fee"], r["shipping_paid"], r["real_price"],
                    r.get("membership_discount", 0), r.get("card_discount", 0)}
        amounts |= {a["amount"] for a in r.get("applied", [])}
    for h in state.get("price_histories") or []:
        amounts |= {h["current_price"], h["min_price"], h["max_price"], h["avg_price"]}
        amounts |= {pt["price"] for pt in h.get("points", [])}
    return amounts


def guardrail_output_node(state: AgentState) -> dict:
    source = state.get("answer_source", "code")
    links = {p["link"] for p in (state.get("products") or {}).values()}
    links |= {c.get("link") for c in (state.get("chunks") or []) + (state.get("blog_refs") or []) if c.get("link")}
    v = guardrails.check_output(state.get("answer", ""), allowed_links=links,
                                allowed_amounts=_allowed_amounts(state) if source == "llm" else None)
    step = {"step": "guardrail_output", "input": f"응답 초안 ({source})", "output": v.trace_output()}
    return {"answer": v.text, "trace": [step]}


# ─────────────────────────────── 그래프 ───────────────────────────────


def _after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and state.get("llm_turns", 0) < MAX_LLM_TURNS:
        return "tools"
    return "answer"


FINAL_TOOLS = {"retrieve_reviews", "calc_real_price", "query_price_db"}


def _after_tools(state: AgentState) -> str:
    """도구 결과 뒤 다음 노드. 프롬프트가 필요한 도구를 한 턴에 모두 부르게 하므로, 마지막 단계 도구
    (리뷰 검색·실결제가·가격 이력)를 실행한 라운드 뒤에는 LLM 턴 없이 바로 답변한다."""
    if state.get("tool_calls", 0) >= MAX_TOOL_CALLS:
        return "answer"
    last_round = set()
    for m in reversed(state["messages"]):
        if not isinstance(m, ToolMessage):
            break
        last_round.add(m.name)
    if last_round & FINAL_TOOLS:
        return "answer"
    return "agent"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrail_input", guardrail_input_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("answer", answer_node)
    g.add_node("guardrail_output", guardrail_output_node)
    g.add_edge(START, "guardrail_input")
    g.add_conditional_edges("guardrail_input", lambda s: "guardrail_output" if s.get("blocked") else "agent",
                            {"agent": "agent", "guardrail_output": "guardrail_output"})
    g.add_conditional_edges("agent", _after_agent, {"tools": "tools", "answer": "answer"})
    g.add_conditional_edges("tools", _after_tools, {"agent": "agent", "answer": "answer"})
    g.add_edge("answer", "guardrail_output")
    g.add_edge("guardrail_output", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─────────────────────────────── 실행·관측 ───────────────────────────────


def warmup() -> float:
    """임베딩·리랭커 모델과 BM25·협찬 라벨을 미리 올린다 (첫 요청의 모델 로딩 약 18초 제거).
    CLI 시작과 API 서버 시작 시 1회 호출한다."""
    from src import indexer

    t = time.perf_counter()
    indexer.get_embedder().embed_query("warmup")
    retriever.get_reranker().score("warmup", ["warmup"])
    if indexer.get_collection().count():
        indexer.review_sponsor_labels()
    return time.perf_counter() - t


def langsmith_enabled() -> bool:
    from langsmith import utils

    return bool(utils.tracing_is_enabled() and os.getenv("LANGSMITH_API_KEY"))


def fetch_langsmith_metrics(run_id: str, wait_s: float = 20.0) -> dict:
    """LangSmith에 기록된 루트 run의 총 토큰·지연 시간을 조회한다 (업로드 지연을 고려해 재시도)."""
    if not langsmith_enabled():
        return {"enabled": False}
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    import warnings

    wait_for_all_tracers()
    client = Client()
    deadline = time.time() + wait_s
    while True:
        try:
            with warnings.catch_warnings():  # langsmith 베타 API 경고 숨김
                warnings.simplefilter("ignore")
                run = client.read_run(run_id)
                url = client.get_run_url(run=run) if run.end_time is not None and run.total_tokens else ""
            if url:
                return {"enabled": True, "total_tokens": run.total_tokens,
                        "prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
                        "latency_s": round((run.end_time - run.start_time).total_seconds(), 2), "url": url}
        except Exception as exc:  # noqa: BLE001 - 아직 업로드 전이면 404
            last = type(exc).__name__
        else:
            last = "집계 대기"
        if time.time() > deadline:
            return {"enabled": True, "error": f"조회 실패({last})"}
        time.sleep(2)


def run(question: str, with_langsmith: bool = False) -> AgentResult:
    """질문 1건 처리 (무상태). with_langsmith=True면 LangSmith에서 토큰·지연을 조회한다."""
    from langchain_core.callbacks import get_usage_metadata_callback

    run_id = str(uuid.uuid4())
    question = mask_pii(question)   # 사용자 입력 PII는 그래프·LangSmith·로그에 남기지 않는다 (정책 3)
    config = {"run_name": "shopping_agent", "run_id": run_id, "recursion_limit": 40,
              "tags": ["shopping-agent"],
              "metadata": {"review_source": os.getenv("REVIEW_SOURCE", "sample"),
                           "retriever_mode": os.getenv("RETRIEVER_MODE", "vector")}}
    t = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        out = get_graph().invoke({"messages": [HumanMessage(question)], "question": question,
                                  "trace": [], "timings": []}, config=config)
    latency = time.perf_counter() - t
    tokens = {"input": 0, "output": 0, "total": 0}
    for u in cb.usage_metadata.values():
        tokens["input"] += u.get("input_tokens", 0)
        tokens["output"] += u.get("output_tokens", 0)
        tokens["total"] += u.get("total_tokens", 0)
    return AgentResult(answer=out["answer"], contexts=out.get("contexts", []), trace=out.get("trace", []),
                       run_id=run_id, latency_s=round(latency, 2), tokens=tokens,
                       langsmith=fetch_langsmith_metrics(run_id) if with_langsmith else {},
                       timings=[(k, round(v, 2)) for k, v in out.get("timings", [])])


# ─────────────────────────────── CLI ───────────────────────────────


SPEEDUP_TIPS = {
    "answer:review_analysis": "리뷰 분석은 모델 수만큼 병렬 호출이라 가장 느린 1건이 전체 시간이다. 분석 프롬프트 출력(장단점 개수·글자 수)을 더 줄이거나, "
                              "리뷰 분석에 더 빠른 모델(BEDROCK_MODEL_ID와 별도 설정)을 쓰거나, 카드에 오를 상위 3개 모델만 분석한다.",
    "answer:cards": "카드 조립은 코드 계산이라 느리면 use_card_writer=True(카드 문구 LLM 추가 호출)가 켜져 있는지 확인한다.",
    "agent_llm": "에이전트 LLM 턴 수를 줄인다(첫 턴에 search_products와 calc_real_price를 함께 요청, 재검색 최소화). "
                 "도구 스키마·시스템 프롬프트를 줄여 입력 토큰을 낮추거나 에이전트에 더 빠른 모델을 쓴다.",
    "tool:retrieve_reviews": "리랭커(CPU, 후보 20개)가 대부분이다. RERANK_POOL을 줄이거나 서버 시작 시 임베딩·리랭커를 미리 로드한다.",
    "tool:fetch_reviews": "sample 모드는 색인된 리뷰를 건너뛰므로 첫 호출의 모델 로딩이 대부분이다. 서버 시작 시 미리 로드한다.",
}


def _load_rows(ids: list[str]) -> list[dict]:
    with QUERIES_PATH.open(encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    return [rows[i] for i in ids]


def _print_result(label: str, r: AgentResult, expected_tools: list[str] | None = None) -> None:
    print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
    print(r.answer)
    print("\n── trace ──")
    for s in r.trace:
        print(f"  [{s['step']}] {s['input']}  →  {s['output']}")
    if expected_tools is not None:
        called = {s["step"] for s in r.trace}
        missing = [t for t in expected_tools if t not in called]
        print(f"  expected_tools {expected_tools} → {'모두 호출' if not missing else f'누락 {missing}'}")
    print(f"\n── contexts: {len(r.contexts)}개 {[c['doc_id'] for c in r.contexts]}")
    print("── 단계별 시간 ──")
    for k, v in r.timings:
        print(f"  {k:<28} {v:6.2f}초")
    ls = r.langsmith
    ls_text = (f"LangSmith 총 토큰 {ls.get('total_tokens'):,} (입력 {ls.get('prompt_tokens'):,} / 출력 "
               f"{ls.get('completion_tokens'):,}) · 지연 {ls.get('latency_s')}초" if ls.get("total_tokens")
               else f"LangSmith {ls.get('error') or '꺼짐'}")
    print(f"── 응답 시간 {r.latency_s}초 · 로컬 집계 토큰 {r.tokens['total']:,} (입력 {r.tokens['input']:,} / "
          f"출력 {r.tokens['output']:,}) · {ls_text}")
    if r.latency_s > RESPONSE_BUDGET_S and r.timings:
        slow_k, slow_v = max(r.timings, key=lambda kv: kv[1])
        llm_total = sum(v for k, v in r.timings if k.startswith("agent_llm"))
        print(f"⚠ 30초 초과: 가장 오래 걸린 단계 {slow_k} {slow_v:.1f}초 (에이전트 LLM 턴 합계 {llm_total:.1f}초)")
        key = "agent_llm" if slow_k.startswith("agent_llm") else slow_k
        print(f"  개선 방법: {SPEEDUP_TIPS.get(key, '해당 단계 입력·출력 크기를 줄인다.')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="쇼핑 비교 ReAct 에이전트 CLI")
    ap.add_argument("question", nargs="?", help="질문")
    ap.add_argument("--ids", help="evaluation/test_queries.csv 문항 id (쉼표 구분)")
    ap.add_argument("--no-langsmith", action="store_true", help="LangSmith 조회 생략")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    on = langsmith_enabled()
    print(f"LangSmith 트레이싱: {'켜짐' if on else '꺼짐'} (프로젝트 {'설정됨' if os.getenv('LANGSMITH_PROJECT') else '기본'})"
          f" · REVIEW_SOURCE={os.getenv('REVIEW_SOURCE', 'sample')} · RETRIEVER_MODE={os.getenv('RETRIEVER_MODE', 'vector')}")
    with_ls = on and not args.no_langsmith
    print(f"모델 예열 {warmup():.1f}초 (응답 시간에서 제외, API는 서버 시작 시 1회)")
    if args.ids:
        rows = _load_rows([i.strip() for i in args.ids.split(",") if i.strip()])
        results = []
        for row in rows:
            r = run(row["input"], with_langsmith=with_ls)
            exp = [t for t in row["expected_tools"].split(";") if t]
            _print_result(f"#{row['id']} [{row['category']}] {row['input']}", r, exp)
            results.append((row["id"], r))
        print(f"\n{'=' * 100}\n요약")
        for qid, r in results:
            print(f"  #{qid}: {r.latency_s}초 · 토큰 {r.langsmith.get('total_tokens') or r.tokens['total']:,}"
                  f" · {'30초 초과' if r.latency_s > RESPONSE_BUDGET_S else '30초 이내'}")
    elif args.question:
        _print_result(args.question, run(args.question, with_langsmith=with_ls))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
