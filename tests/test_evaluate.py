"""evaluate.py의 순수 함수 테스트 (LLM·API 호출 없음)."""

from src import evaluate as ev

ROW = {"id": "1", "category": "positive", "input": "q",
       "traits": ["서로 다른 모델 3개 제시", "출처 링크"],
       "forbidden_items": ["35만원 초과 상품", "같은 모델 중복 추천", "전화번호 형식 문자열"],
       "tools": ["search_products", "calc_real_price"]}


def _resp(answer, steps=("guardrail_input", "search_products", "calc_real_price", "answer", "guardrail_output")):
    return {"answer": answer, "contexts": [], "trace": [{"step": s, "input": "", "output": ""} for s in steps]}


def _judge(met=(True, True), found=(False, False, False), score=5, ftype="없음"):
    return {"traits": [{"trait": t, "met": m, "reason": ""} for t, m in zip(ROW["traits"], met)],
            "forbidden": [{"item": f, "found": x, "reason": ""} for f, x in zip(ROW["forbidden_items"], found)],
            "score": score, "score_reason": "", "failure_type": ftype}


def test_model_family_ignores_profile_and_version():
    assert ev.model_family("us.anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert ev.model_family("global.anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"   # 같은 모델로 판정
    assert ev.model_family("us.anthropic.claude-sonnet-4-5-20250929-v1:0") == "claude-sonnet-4-5"


def test_rule_checks_budget_duplicate_pii_tools():
    ans = ("1) A모델 · 최저 실결제가\n   - 몰: 판매가 360,000원 + 배송비 0원\n"
           "2) A모델 · 최저\n연락처 010-1234-5678")
    r = ev.rule_checks(ROW, _resp(ans, steps=("guardrail_input", "search_products", "answer")))
    assert r["tools_missing"] == ["calc_real_price"]
    assert any("35만원 초과" in f for f in r["forbidden_rule"])
    assert "같은 모델 중복 추천" in r["forbidden_rule"] and "개인정보 형식 문자열" in r["forbidden_rule"]
    ok = ev.rule_checks(ROW, _resp("1) A · x\n   - 몰: 판매가 349,000원\n2) B · y"))
    assert ok == {"tools_missing": [], "forbidden_rule": []}


def test_verdict_requires_all_three_conditions():
    good = {"call": {"status": 200, "latency_s": 1.0, "response": _resp("1) A · x\n2) B · y")}, "judge": _judge()}
    assert ev.verdict(ROW, good)["passed"]
    trait_fail = {**good, "judge": _judge(met=(True, False), ftype="형식·안내 누락")}
    v = ev.verdict(ROW, trait_fail)
    assert not v["passed"] and v["failure_type"] == "형식·안내 누락" and v["traits_met"] == 1
    tool_fail = {**good, "call": {**good["call"], "response": _resp("x", steps=("guardrail_input", "answer"))}}
    assert ev.verdict(ROW, tool_fail)["failure_type"] == "도구 누락"
    api_fail = {**good, "call": {**good["call"], "status": 503}}
    assert ev.verdict(ROW, api_fail)["failure_type"] == "API 오류"


def test_ragas_eligibility_reasons():
    clar = {"call": {"status": 200, "response": _resp("예산은?", steps=("guardrail_input", "answer"))}}
    ok, why = ev.ragas_eligible({"category": "edge"}, clar)
    assert not ok and "되묻기" in why
    price = {"call": {"status": 200, "response": _resp("가격")}}
    assert "리뷰 검색" in ev.ragas_eligible({"category": "positive"}, price)[1]
    assert ev.ragas_eligible({"category": "guardrail"}, price)[1] == "positive·edge가 아님"
    rag = {"call": {"status": 200, "response": {**_resp("a"), "contexts": [{"doc_id": "rv-1#c0", "text": "t"}]}}}
    assert ev.ragas_eligible({"category": "positive"}, rag) == (True, "")
