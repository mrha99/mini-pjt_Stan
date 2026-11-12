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


CARD_ANSWER = """조건에 맞는 무선청소기 2개를 추천합니다. (샘플 데이터)

1) 클린웨이브 V7 사일런트 무선청소기 · 최저 실결제가 242,000원(바로마켓, 배송비 포함) · 판매처 3곳 비교
   요약: 저소음이라 원룸 야간 청소에 적합
   판매처별 실결제가 [샘플 데이터]:
   - 바로마켓: 판매가 239,000원 + 배송비 3,000원 = 242,000원 · https://x
   추천 근거:
   - 밤에도 옆집 눈치 안 보일 만큼 조용함 [1][2]
   - 최저 실결제가 242,000원(바로마켓)
   단점(실구매 후기): 먼지통이 작음 [1]; 강모드 배터리가 짧음 [2]
   이런 분께는 비추천: 반려동물 털이 많은 집
   리뷰 구분: 이 모델 리뷰 11건 중 협찬 표시 2건은 근거에서 제외

2) 미라홈 슬림 S3 무선청소기 · 최저 실결제가 191,500원(온새몰, 배송비 포함) · 판매처 1곳 비교
   요약: 실구매 후기 없음 · 가격 기준으로만 포함
   추천 근거:
   - 최저 실결제가 191,500원(온새몰)

출처 (실구매 후기):
[1] 후기 · https://blog.example.com/a"""


def test_review_response_keeps_only_review_sentences_with_model():
    lines = ev.review_response(CARD_ANSWER).splitlines()
    assert lines == ["클린웨이브 V7 사일런트 무선청소기: 저소음이라 원룸 야간 청소에 적합",
                     "클린웨이브 V7 사일런트 무선청소기: 밤에도 옆집 눈치 안 보일 만큼 조용함",
                     "클린웨이브 V7 사일런트 무선청소기: 먼지통이 작음",
                     "클린웨이브 V7 사일런트 무선청소기: 강모드 배터리가 짧음",
                     "클린웨이브 V7 사일런트 무선청소기: 비추천 대상 - 반려동물 털이 많은 집"]


def test_review_response_summary_format_and_free_text():
    summary = "실구매 후기 요약입니다.\n\n■ 에어리움 터보 X9\n   요약: 흡입력 우수\n   장점:\n   - 털 엉킴 없음 [1]\n   단점:\n   - 무거움 [2]"
    assert ev.review_response(summary).splitlines() == ["에어리움 터보 X9: 흡입력 우수", "에어리움 터보 X9: 털 엉킴 없음",
                                                         "에어리움 터보 X9: 무거움"]
    assert ev.review_response("해당 상품은 카탈로그에 없습니다.") == "해당 상품은 카탈로그에 없습니다."


def test_review_traits_filter():
    traits = ["서로 다른 모델 3개 제시", "소음 관련 실구매 리뷰 근거", "출처 링크", "원룸에서의 소음 단점 언급",
              "실구매 후기가 없고 협찬 리뷰만 있다고 안내"]
    assert ev.review_traits(traits) == ["소음 관련 실구매 리뷰 근거", "원룸에서의 소음 단점 언급",
                                        "실구매 후기가 없고 협찬 리뷰만 있다고 안내"]


def test_migrate_ragas_moves_unversioned_values(tmp_path):
    store = ev.Store(tmp_path / "r.json", fresh=True)
    store.item("1")["ragas"] = {"faithfulness": 0.1}
    store.item("2").update({"ragas": {"faithfulness": 0.9}, "ragas_version": "review"})
    ev.migrate_ragas(store)
    assert store.item("1") == {"ragas_orig": {"faithfulness": 0.1}}
    assert store.item("2")["ragas"] == {"faithfulness": 0.9}
