"""chains.py 테스트 (LLM 호출 없음: 가짜 LLM 주입)."""

import threading
import time

import pytest
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableLambda

from src import chains
from src.chains import (CardText, CardTexts, ConditionPoint, ReasonItem, ReviewAnalysis,
                        ReviewVerdict, SponsorshipJudgment, classify_rule)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Converse")


class FakeLLM:
    """with_structured_output(schema) → schema별 응답 함수를 부르는 Runnable."""

    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        def run(prompt_value):
            self.calls.append(schema.__name__)
            return self.handlers[schema.__name__](prompt_value.to_string())
        return RunnableLambda(run)


@pytest.fixture
def fake_llm():
    def install(handlers):
        llm = FakeLLM(handlers)
        chains.set_llm(llm)
        return llm
    yield install
    chains.set_llm(None)


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch):
    monkeypatch.setattr(chains, "LLM_RETRY_BASE_DELAY", 0.001)


# ─────────────────────────────── 규칙 ───────────────────────────────


@pytest.mark.parametrize("text", [
    "누리쇼핑에서 할인할 때 샀고 협찬 아닙니다.",
    "광고 아니고 직접 결제해서 산 거예요.",
    "협찬 리뷰가 너무 많아서 직접 구매해 봤어요.",
    "협찬받지 않았어요. 솔직 후기",
    "협찬 X 내돈내산",
])
def test_rule_negation_is_not_sponsored(text):
    assert classify_rule(text).label == "실구매"


@pytest.mark.parametrize("text", [
    "[협찬] 제품을 대여받아 2주간 사용한 뒤 작성한 글입니다.",
    "이 리뷰는 업체로부터 제품을 무상 제공받아 작성되었습니다.",
    "본 포스팅은 소정의 원고료를 지원받아 작성되었습니다.",
    "#협찬 #체험단",
    "미라홈 케어 40 2주 사용기 (제품 제공)",
    "협찬 아닙니다. 사실 체험단으로 받은 제품이에요.",  # 부정 뒤에 다른 대가성 문구
])
def test_rule_disclosure_is_sponsored(text):
    assert classify_rule(text).label == "협찬"


def test_rule_ad_signals_go_to_llm():
    r = classify_rule("거치대가 작아요. 프로필 링크에서 할인코드 V719 쓰면 추가 할인돼요")
    assert r.label is None and "할인코드" in r.ad_signals
    assert classify_rule("직접 구매했어요. 구매 링크는 하단에").label is None


def test_rule_never_wrong_on_labels():
    """규칙이 판정한 라벨 리뷰는 모두 정답이어야 한다 (미정은 LLM 몫)."""
    for row in chains._load_labeled():
        r = classify_rule(f"{row['title']}\n{row['body']}")
        assert r.label in (None, row["gold"]), row["review_id"]


# ─────────────────────────────── 재시도·동시성 ───────────────────────────────


def test_retry_on_throttling_then_success():
    n = {"calls": 0}

    def flaky(_):
        n["calls"] += 1
        if n["calls"] < 3:
            raise _client_error("ThrottlingException")
        return "ok"

    assert chains._guarded(RunnableLambda(flaky), "t").invoke({}) == "ok"
    assert n["calls"] == 3


def test_no_retry_on_validation_error():
    n = {"calls": 0}

    def bad(_):
        n["calls"] += 1
        raise _client_error("ValidationException")

    with pytest.raises(ClientError):
        chains._guarded(RunnableLambda(bad), "t").invoke({})
    assert n["calls"] == 1


def test_gives_up_after_max_attempts():
    n = {"calls": 0}

    def always(_):
        n["calls"] += 1
        raise _client_error("ThrottlingException")

    with pytest.raises(ClientError):
        chains._guarded(RunnableLambda(always), "t").invoke({})
    assert n["calls"] == chains.LLM_MAX_ATTEMPTS


def test_empty_structured_output_is_retried():
    outs = iter([None, "ok"])
    assert chains._guarded(RunnableLambda(lambda _: next(outs)), "t").invoke({}) == "ok"


def test_concurrency_limit(monkeypatch):
    monkeypatch.setattr(chains, "_LLM_SEMAPHORE", threading.BoundedSemaphore(2))
    lock, state = threading.Lock(), {"now": 0, "max": 0}

    def slow(x):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return x

    out = chains._guarded(RunnableLambda(slow), "t").batch([{}] * 8, config={"max_concurrency": 8})
    assert len(out) == 8 and state["max"] <= 2


# ─────────────────────────────── 협찬 판별 (규칙 → LLM) ───────────────────────────────


def test_classify_reviews_llm_only_for_undecided(fake_llm):
    llm = fake_llm({"SponsorshipJudgment": lambda p: SponsorshipJudgment(
        label="협찬", reason="할인코드 홍보", confidence="high")})
    reviews = [
        {"review_id": "a", "title": "", "body": "내돈내산 후기"},
        {"review_id": "b", "title": "", "body": "#협찬 받았어요"},
        {"review_id": "c", "title": "", "body": "프로필 링크 할인코드 쓰세요"},
    ]
    v = chains.classify_reviews(reviews)
    assert [x.label for x in v] == ["실구매", "협찬", "협찬"]
    assert [x.method for x in v] == ["rule", "rule", "llm"]
    assert llm.calls == ["SponsorshipJudgment"]  # 미정 1건만 LLM


def test_low_confidence_becomes_unknown(fake_llm):
    fake_llm({"SponsorshipJudgment": lambda p: SponsorshipJudgment(
        label="실구매", reason="애매", confidence="low")})
    v = chains.classify_reviews([{"review_id": "c", "title": "", "body": "좋아요"}])
    assert v[0].label == "판별 불가"


# ─────────────────────────────── 모델 분석·카드·렌더링 ───────────────────────────────


def _chunk(doc_id, model_id, text="본문", link=None):
    rid = doc_id.split("#")[0]
    return {"doc_id": doc_id, "review_id": rid, "model_id": model_id, "text": text, "source": "sample",
            "title": f"{rid} 제목", "link": link or f"https://blog.example.com/{rid}"}


@pytest.fixture
def reviews(monkeypatch):
    data = {
        "rv-a": {"review_id": "rv-a", "title": "내돈내산", "body": "직접 구매. 조용해요"},
        "rv-b": {"review_id": "rv-b", "title": "[협찬]", "body": "원고료 받았어요. 최고"},
        "rv-c": {"review_id": "rv-c", "title": "후기", "body": "프로필 링크 할인코드"},
        "rv-d": {"review_id": "rv-d", "title": "내돈내산", "body": "직접 구매. 시끄러워요"},
        "rv-e": {"review_id": "rv-e", "title": "내돈내산", "body": "직접 구매. 가벼워요"},
    }
    monkeypatch.setattr(chains, "review_text", data.get)
    monkeypatch.setattr(chains, "_models", lambda: {
        "M1": {"name": "모델1", "brand": "가"}, "M2": {"name": "모델2", "brand": "나"},
        "M3": {"name": "모델3", "brand": "다"}, "M4": {"name": "모델4", "brand": "라"}})


def _analysis_handler(prompt: str) -> ReviewAnalysis:
    mid = "M1" if "M1" in prompt else "M2"
    return ReviewAnalysis(
        model_id=mid,
        # LLM이 규칙 판정(rv-b 협찬)을 뒤집으려 해도 코드가 규칙을 우선한다
        verdicts=[ReviewVerdict(review_id="rv-b", label="실구매", reason="x"),
                  ReviewVerdict(review_id="rv-c", label="협찬", reason="할인코드")],
        pros=[ConditionPoint(condition="소음", point="조용함", chunk_ids=["rv-a#c0", "rv-b#c0"]),
              ConditionPoint(condition="일반", point="지어낸 장점", chunk_ids=["rv-zzz#c9"])],
        cons=[ConditionPoint(condition="소음", point="시끄러움", chunk_ids=["rv-d#c0"])],
        not_recommended_for=["예민한 사람"], evidence_chunk_ids=["rv-a#c0", "rv-c#c0"], summary="조용한 편",
    )


def test_analyze_models_one_call_per_model_and_postprocess(fake_llm, reviews):
    llm = fake_llm({"ReviewAnalysis": _analysis_handler})
    chunks = [_chunk("rv-a#c0", "M1"), _chunk("rv-b#c0", "M1"), _chunk("rv-c#c0", "M1"),
              _chunk("rv-d#c0", "M1"), _chunk("rv-e#c0", "M2")]
    out = chains.analyze_models("조용한 청소기", chunks, ["M1", "M2", "M3"], ["소음"])
    assert llm.calls.count("ReviewAnalysis") == 2           # 청크 있는 모델당 1회, M3는 호출 없음
    a = out["M1"]
    labels = {v.review_id: (v.label, v.method) for v in a.verdicts}
    assert labels["rv-b"] == ("협찬", "rule") and labels["rv-c"] == ("협찬", "llm")
    assert labels["rv-a"] == ("실구매", "rule")
    assert [p.chunk_ids for p in a.pros] == [["rv-a#c0"]]    # 협찬 청크·없는 청크 제거
    assert "rv-c#c0" not in a.evidence_chunk_ids
    assert out["M3"].verdicts == [] and out["M3"].pros == []


def test_analyze_models_llm_failure_falls_back(fake_llm, reviews):
    def boom(_):
        raise _client_error("AccessDeniedException")
    fake_llm({"ReviewAnalysis": boom})
    out = chains.analyze_models("q", [_chunk("rv-a#c0", "M1"), _chunk("rv-c#c0", "M1")], ["M1"])
    labels = {v.review_id: v.label for v in out["M1"].verdicts}
    assert labels == {"rv-a": "실구매", "rv-c": "판별 불가"}
    assert out["M1"].error == "LLM 오류 ClientError"            # 실패 사유는 trace·로그용으로 남는다
    assert "error" not in out["M1"].model_dump()                  # 스키마·출력에는 없음


def _product(pid, mid, mall, price, ship=0):
    return {"product_id": pid, "model_id": mid, "mall": mall, "price": price, "shipping_fee": ship,
            "link": f"https://{mall}.example.com/{pid}"}


def _real(pid, price, real, ship=0):
    return {"product_id": pid, "price": price, "shipping_paid": ship, "real_price": real,
            "membership_discount": 0, "card_discount": price + ship - real,
            "applied": [{"kind": "card", "name": "한결카드", "detail": "5%", "amount": price + ship - real}] if real < price + ship else []}


def _cards_fixture(fake_llm, reviews, card_handler, use_card_writer=False):
    llm = fake_llm({"CardTexts": card_handler})
    products = [_product("P1", "M1", "하늘몰", 300000), _product("P2", "M1", "바로마켓", 290000, 3000),
                _product("P3", "M2", "하늘몰", 200000), _product("P4", "M3", "모아딜", 250000),
                _product("P5", "M4", "모아딜", 400000)]
    reals = [_real("P1", 300000, 285000), _real("P2", 290000, 293000, 3000), _real("P3", 200000, 200000),
             _real("P4", 250000, 250000), _real("P5", 400000, 400000)]
    pros = [ConditionPoint(condition="소음", point="조용함", chunk_ids=["rv-a#c0"])]
    analyses = {
        "M1": ReviewAnalysis(model_id="M1", verdicts=[ReviewVerdict(review_id="rv-a", label="실구매", reason="r"),
                                                      ReviewVerdict(review_id="rv-b", label="협찬", reason="r")],
                             pros=pros, evidence_chunk_ids=["rv-a#c0"], summary="조용",
                             reasons=[ReasonItem(text="원룸에서 조용함", chunk_ids=["rv-a#c0", "rv-x#c1"])]),
        "M2": ReviewAnalysis(model_id="M2", verdicts=[], summary="실구매 후기 근거 부족"),
        "M3": ReviewAnalysis(model_id="M3", verdicts=[], summary="실구매 후기 근거 부족"),
        "M4": ReviewAnalysis(model_id="M4", verdicts=[], summary="실구매 후기 근거 부족"),
    }
    chunks = [_chunk("rv-a#c0", "M1")]
    cards = chains.build_cards("조용한 청소기", products, reals, analyses, chunks, ["소음"],
                               use_card_writer=use_card_writer)
    return cards, llm


def test_build_cards_top3_distinct_models_sorted_malls(fake_llm, reviews):
    def card_handler(p):
        return CardTexts(cards=[CardText(model_id="M1", summary="조용한 모델",
                                         reasons=[ReasonItem(text="조용함", chunk_ids=["rv-a#c0", "rv-x#c1"])])])
    cards, llm = _cards_fixture(fake_llm, reviews, card_handler, use_card_writer=True)
    assert llm.calls == ["CardTexts"]
    assert len(cards) == 3 and len({c.model_id for c in cards}) == 3
    assert "M4" not in {c.model_id for c in cards}          # 가장 비싸고 근거 없음
    m1 = next(c for c in cards if c.model_id == "M1")
    assert [p.mall for p in m1.prices] == ["하늘몰", "바로마켓"]  # 실결제가 오름차순
    assert m1.reasons[0].chunk_ids == ["rv-a#c0"]           # 없는 청크 제거
    assert m1.sponsorship.organic == 1 and m1.sponsorship.sponsored == 1
    assert m1.sources[0].review_id == "rv-a"
    other = next(c for c in cards if c.model_id == "M2")
    assert other.reasons and "실결제가" in other.reasons[-1].text  # 문구 누락 모델은 fallback


def test_build_cards_writer_failure_uses_fallback(fake_llm, reviews):
    def boom(_):
        raise _client_error("ValidationException")
    cards, _ = _cards_fixture(fake_llm, reviews, boom, use_card_writer=True)
    assert all(c.reasons for c in cards)


def test_build_cards_default_uses_analysis_text_without_extra_llm_call(fake_llm, reviews):
    cards, llm = _cards_fixture(fake_llm, reviews, lambda p: CardTexts(cards=[]))
    assert llm.calls == []                                       # 카드 문구 전용 호출 없음
    m1 = next(c for c in cards if c.model_id == "M1")
    assert m1.summary == "조용"
    assert m1.reasons[0].text == "원룸에서 조용함" and m1.reasons[0].chunk_ids == ["rv-a#c0"]
    assert m1.reasons[-1].text.startswith("최저 실결제가 285,000원")


def test_render_answer(fake_llm, reviews):
    cards, _ = _cards_fixture(fake_llm, reviews, lambda p: CardTexts(cards=[]))
    text = chains.render_answer(cards[:2], category="무선청소기",
                                blog_refs=[{"title": "블로그 글", "link": "https://blog.naver.com/x/1"}])
    assert "샘플 데이터" in text and "판매처별 실결제가 [샘플 데이터]" in text
    assert "모델이 2개뿐" in text
    assert "협찬 1건" in text and "[1] rv-a 제목" in text
    assert "실제 블로그 참고 후기 (특정 상품의 근거가 아님)" in text
    assert "285,000원(하늘몰" in text
    assert chains.render_answer([]).startswith("조건에 맞는 상품을 찾지 못했습니다")


def test_cards_evidence_first_and_review_stats(fake_llm, reviews, monkeypatch):
    from collections import Counter
    monkeypatch.setattr(chains, "_model_review_stats",
                        lambda: {"M1": Counter({"실구매": 3, "협찬": 2, "미정": 1})})
    cards, _ = _cards_fixture(fake_llm, reviews, lambda p: CardTexts(cards=[
        CardText(model_id="M2", summary="LLM이 쓴 요약", reasons=[ReasonItem(text="쌈")])]), use_card_writer=True)
    assert cards[0].model_id == "M1" and cards[0].has_evidence   # 더 비싸도 근거 있는 모델이 1위
    no_ev = [c for c in cards if not c.has_evidence]
    assert no_ev and all(c.summary == chains.NO_EVIDENCE_NOTE for c in no_ev)
    sp = cards[0].sponsorship
    assert (sp.total_reviews, sp.disclosed_sponsored) == (6, 2)
    text = chains.render_answer(cards)
    assert "이 모델 리뷰 6건 중 협찬 표시 2건은 근거에서 제외" in text
    assert "실구매 후기 없음 · 가격 기준으로만 포함" in text


def test_finalize_replaces_contradictory_summary(reviews):
    rules = chains._rule_verdicts(["rv-a"], [_chunk("rv-a#c0", "M1")])
    raw = ReviewAnalysis(model_id="M1", verdicts=[], summary="실구매 후기 근거 부족",
                         pros=[ConditionPoint(condition="소음", point="조용함", chunk_ids=["rv-a#c0"])],
                         reasons=[ReasonItem(text="조용", chunk_ids=["rv-a#c0"]),
                                  ReasonItem(text="지어낸 근거", chunk_ids=["rv-b#c0"])])
    a = chains.finalize_analysis(raw, "M1", rules, [_chunk("rv-a#c0", "M1")])
    assert a.summary == "조용함"
    assert [r.text for r in a.reasons] == ["조용"]          # 실구매 청크 근거 없는 추천 근거 제거
