"""도구 5종 테스트. 외부 HTTP는 MockTransport로 대체해 API 키 없이 통과한다."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from src import seed_db, tools
from src.tools import (
    BLOG_LABEL,
    calc_real_price,
    calc_real_price_impl,
    clean_snippet,
    fetch_reviews,
    query_price_db,
    retrieve_reviews,
    search_products,
)

VC_M1_NAME = "클린웨이브 V7 사일런트 무선청소기"  # VC-M1: VC-001 하늘몰, VC-002 바로마켓, VC-003 모아딜(리뷰 없음)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """임시 가격 DB (seed_db로 생성). 실제 data/prices.db를 건드리지 않는다."""
    db = tmp_path / "prices.db"
    seed_db.seed(db_path=db)
    monkeypatch.setattr(tools, "DB_PATH", db)
    return db


@pytest.fixture
def no_sleep(monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: delays.append(s))
    return delays


# ─────────────────────────────── search_products ───────────────────────────────


def test_search_products_fields_and_price_range():
    out = search_products.invoke({"keyword": "무선청소기", "max_price": 350000})
    assert out["count"] > 0 and out["notice"].startswith("샘플 데이터")
    for p in out["products"]:
        assert p["category"] == "무선청소기"
        assert p["price"] <= 350000
        for key in ("product_id", "model_id", "name", "mall", "price", "shipping_fee",
                    "card_discounts", "membership_discounts", "link"):
            assert key in p
    # 같은 모델의 여러 판매처가 함께 나온다 (6단계 카드 묶기용)
    malls = {p["mall"] for p in out["products"] if p["model_id"] == "VC-M1"}
    assert malls == {"하늘몰", "바로마켓", "모아딜"}
    assert out["model_ids"] == list(dict.fromkeys(p["model_id"] for p in out["products"]))


def test_search_products_min_price_and_limit():
    out = search_products.invoke({"keyword": "청소기", "min_price": 300000, "limit": 2})
    assert out["count"] <= 2
    assert all(p["price"] >= 300000 for p in out["products"])


def test_search_products_unknown_returns_empty():
    out = search_products.invoke({"keyword": "XZ-9999"})
    assert out["count"] == 0 and out["products"] == []


def test_search_products_source_is_swappable():
    class FakeSource:
        def search(self, keyword, min_price, max_price, limit):
            return [{"product_id": "X-1", "model_id": "X-M", "name": "가짜", "brand": "B",
                     "category": "C", "mall": "몰", "price": 1000, "shipping_fee": 0,
                     "card_discounts": [], "membership_discounts": [], "link": "https://e.example.com"}]

        def find_by_name(self, name):
            return None

        def list_by_model(self, model_id):
            return []

    original = tools.get_product_source()
    tools.set_product_source(FakeSource())
    try:
        out = search_products.invoke({"keyword": "아무거나"})
        assert out["products"][0]["product_id"] == "X-1"
    finally:
        tools.set_product_source(original)


# ─────────────────────────────── fetch_reviews: sample ───────────────────────────────


def test_fetch_reviews_sample_collects_by_model(monkeypatch):
    monkeypatch.setenv("REVIEW_SOURCE", "sample")
    out = fetch_reviews.invoke({"product_names": [VC_M1_NAME, "없는상품 ZZ-1"]})
    assert out["source"] == "sample"
    assert out["model_ids"] == ["VC-M1"]
    assert out["not_found"] == ["없는상품 ZZ-1"]

    data = json.loads(tools.REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]
    expected = {r["review_id"] for r in data if r["product_id"] in {"VC-001", "VC-002", "VC-003"}}
    got = {r["review_id"] for r in out["reviews"]}
    assert got == expected and len(got) > 0
    # 여러 판매처 리뷰가 모델 단위로 모인다
    assert {r["product_id"] for r in out["reviews"]} >= {"VC-001", "VC-002"}
    assert all(r["model_id"] == "VC-M1" and r["source"] == "sample" for r in out["reviews"])


def test_fetch_reviews_sample_dedupes_same_model_and_reports_no_review(monkeypatch):
    monkeypatch.setenv("REVIEW_SOURCE", "sample")
    # 같은 모델을 두 번(상품명·product_id) 넘겨도 한 번만 모은다
    once = fetch_reviews.invoke({"product_names": [VC_M1_NAME]})
    twice = fetch_reviews.invoke({"product_names": [VC_M1_NAME, "VC-002"]})
    assert twice["count"] == once["count"]

    # 리뷰가 하나도 없는 모델은 no_review_models로 알린다
    reviews = json.loads(tools.REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]
    reviewed = {r["product_id"] for r in reviews}
    products = json.loads(tools.PRODUCTS_PATH.read_text(encoding="utf-8"))["products"]
    by_model: dict[str, list[str]] = {}
    for p in products:
        by_model.setdefault(p["model_id"], []).append(p["product_id"])
    empty_models = [m for m, pids in by_model.items() if not reviewed & set(pids)]
    assert empty_models, "샘플 데이터에 리뷰 없는 모델 픽스처가 있어야 한다"
    out = fetch_reviews.invoke({"product_names": [empty_models[0]]})
    assert out["no_review_models"] == [empty_models[0]] and out["count"] == 0


# ─────────────────────────────── fetch_reviews: blog (mock) ───────────────────────────────


BLOG_RESPONSE = {
    "items": [
        {"title": "원룸 <b>무선청소기</b> 소음 &quot;후기&quot;",
         "link": "https://blog.naver.com/someone/223456789",
         "description": "밤에 돌려도 &lt;조용&gt; 해요 &amp; 만족. 문의 010-1234-5678",
         "bloggername": "살림<b>러</b>", "postdate": "20261101"},
        {"title": "두 번째", "link": "https://blog.naver.com/other/223000111",
         "description": "흡입력 <b>좋음</b>", "bloggername": "b", "postdate": "20261020"},
    ]
}


def _mock_client(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def blog_env(monkeypatch, tmp_db, no_sleep):
    monkeypatch.setenv("REVIEW_SOURCE", "blog")
    monkeypatch.setenv("NAVER_CLIENT_ID", "test-id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "test-secret")
    return no_sleep


def test_clean_snippet_strips_tags_and_entities():
    assert clean_snippet("<b>무선</b> &quot;청소기&quot; &amp; &lt;b&gt;") == '무선 "청소기" & <b>'


def test_fetch_reviews_blog_success(blog_env, monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=BLOG_RESPONSE)

    monkeypatch.setattr(tools, "_http_client", _mock_client(handler))
    out = fetch_reviews.invoke({"product_names": [VC_M1_NAME], "conditions": ["원룸", "소음"]})

    req = requests[0]
    assert req.url.host == "naverapihub.apigw.ntruss.com" and req.url.path == "/search/v1/blog"
    assert req.headers["X-NCP-APIGW-API-KEY-ID"] == "test-id"
    assert req.headers["X-NCP-APIGW-API-KEY"] == "test-secret"
    # 가상 상품명 대신 "카테고리 + 조건어 + 후기"로 검색
    assert req.url.params["query"] == "무선청소기 원룸 소음 후기"

    assert out["source"] == "blog" and out["from_cache"] is False and out["count"] == 2
    r0 = out["reviews"][0]
    assert r0["review_id"] == "blog-223456789"
    assert r0["label"] == BLOG_LABEL and r0["model_id"] is None  # 특정 상품 근거로 쓰지 않음
    assert r0["title"] == '원룸 무선청소기 소음 "후기"'
    assert "<" not in r0["title"] and "&quot;" not in r0["title"]
    assert "010-1234-5678" not in r0["body"] and "[전화번호]" in r0["body"]
    assert r0["author"] == "살림러"
    assert BLOG_LABEL in out["notice"]


def test_fetch_reviews_blog_one_call_per_category(blog_env, monkeypatch):
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params["query"])
        return httpx.Response(200, json={"items": []})

    monkeypatch.setattr(tools, "_http_client", _mock_client(handler))
    products = json.loads(tools.PRODUCTS_PATH.read_text(encoding="utf-8"))["products"]
    vc = [p["name"] for p in products if p["category"] == "무선청소기"][:3]
    ap = [p["name"] for p in products if p["category"] == "공기청정기"][:1]
    fetch_reviews.invoke({"product_names": vc + ap})
    assert calls == ["무선청소기 후기", "공기청정기 후기"]


def test_fetch_reviews_blog_retries_then_uses_cache(blog_env, monkeypatch):
    # 1) 성공 호출로 캐시를 채운다
    monkeypatch.setattr(tools, "_http_client",
                        _mock_client(lambda r: httpx.Response(200, json=BLOG_RESPONSE)))
    first = fetch_reviews.invoke({"product_names": [VC_M1_NAME], "conditions": ["소음"]})
    assert first["from_cache"] is False

    # 2) 이후 계속 503 → 3회 시도(지수 백오프) 후 캐시로 대체
    attempts = {"n": 0}

    def failing(request):
        attempts["n"] += 1
        return httpx.Response(503)

    monkeypatch.setattr(tools, "_http_client", _mock_client(failing))
    out = fetch_reviews.invoke({"product_names": [VC_M1_NAME], "conditions": ["소음"]})
    assert attempts["n"] == 3
    assert blog_env == [0.5, 1.0]  # 지수 백오프 대기
    assert out["from_cache"] is True and out["cached_at"]
    assert out["count"] == first["count"]
    assert "캐시" in out["notice"]


def test_fetch_reviews_blog_fail_without_cache(blog_env, monkeypatch):
    monkeypatch.setattr(tools, "_http_client", _mock_client(lambda r: httpx.Response(500)))
    out = fetch_reviews.invoke({"product_names": [VC_M1_NAME], "conditions": ["새조건"]})
    assert out["count"] == 0 and out["from_cache"] is False
    assert "캐시 없음" in out["notice"]


def test_fetch_reviews_blog_without_keys_falls_back(blog_env, monkeypatch):
    monkeypatch.delenv("NAVER_CLIENT_ID")
    called = {"n": 0}

    def handler(request):
        called["n"] += 1
        return httpx.Response(200, json=BLOG_RESPONSE)

    monkeypatch.setattr(tools, "_http_client", _mock_client(handler))
    out = fetch_reviews.invoke({"product_names": [VC_M1_NAME]})
    assert called["n"] == 0 and out["count"] == 0


def test_cache_table_survives_reseed(tmp_db):
    tools._cache_put("k1", "naver_blog", {"items": []})
    seed_db.seed(db_path=tmp_db)  # price_history·meta만 다시 만든다
    assert tools._cache_get("k1") is not None
    with sqlite3.connect(tmp_db) as conn:
        stored = conn.execute("SELECT cache_key FROM api_cache").fetchall()
    assert stored == [("k1",)]


# ─────────────────────────────── retrieve_reviews ───────────────────────────────


def test_retrieve_reviews_empty_index():
    out = retrieve_reviews.invoke({"queries": ["무선청소기 소음"], "model_ids": ["VC-M1"]})
    assert out["status"] == "empty_index" and out["chunks"] == []


def test_retrieve_reviews_calls_retriever(monkeypatch):
    from src import indexer, retriever

    indexer.build_index()
    seen = {}

    def fake_retrieve(queries, model_ids):
        seen.update(queries=queries, model_ids=model_ids)
        return [{"doc_id": "rv-0001#c0", "text": "조용해요", "model_id": "VC-M1", "source": "sample"}]

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    out = retrieve_reviews.invoke({"queries": ["소음"], "model_ids": ["VC-M1"]})
    assert out["status"] == "ok" and out["chunks"][0]["doc_id"] == "rv-0001#c0"
    assert seen == {"queries": ["소음"], "model_ids": ["VC-M1"]}
    assert out["blog_references"] == []  # 블로그 색인이 없으면 비어 있음


def test_retrieve_reviews_has_no_top_k_input():
    # k는 설정값으로 고정 (1·2차 평가 공정 비교)
    assert "top_k" not in retrieve_reviews.args_schema.model_fields


# ─────────────────────────────── query_price_db ───────────────────────────────


def test_query_price_db_history(tmp_db):
    out = query_price_db.invoke({"product_ids": ["VC-001", "NOPE-1"]})
    assert out["not_found"] == ["NOPE-1"]
    h = out["histories"][0]
    assert h["product_id"] == "VC-001" and h["model_id"] == "VC-M1" and h["mall"] == "하늘몰"
    assert len(h["points"]) == 30
    assert h["current_price"] == 249000  # 마지막 날 = products.json 현재가
    assert h["min_price"] <= h["current_price"] <= h["max_price"]
    assert h["points"][0]["date"] < h["points"][-1]["date"]
    assert out["period"] == "2026-10-13 ~ 2026-11-11"
    assert "샘플 데이터" in out["notice"]


def test_query_price_db_days(tmp_db):
    out = query_price_db.invoke({"product_ids": ["VC-001"], "days": 7})
    pts = out["histories"][0]["points"]
    assert len(pts) == 7 and pts[-1]["date"] == "2026-11-11"


def test_query_price_db_missing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DB_PATH", tmp_path / "none.db")
    out = query_price_db.invoke({"product_ids": ["VC-001"]})
    assert out["histories"] == [] and "seed_db" in out["notice"]


# ─────────────────────────────── calc_real_price ───────────────────────────────


def item(**kw):
    base = {"product_id": "T-1", "price": 200000, "shipping_fee": 3000,
            "card_discounts": [], "membership_discounts": []}
    base.update(kw)
    return base


CARD_RATE = {"card": "한결카드", "type": "rate", "value": 0.05, "min_amount": 100000, "max_discount": 15000}


def calc(items, cards=(), memberships=()):
    return calc_real_price_impl(items, list(cards), list(memberships)).results


def test_calc_no_discounts():
    r = calc([item()])[0]
    assert r.real_price == 203000 and r.applied == []


def test_calc_card_not_owned():
    r = calc([item(card_discounts=[CARD_RATE])], cards=["누리카드"])[0]
    assert r.card_discount == 0 and r.real_price == 203000


def test_calc_card_rate_applied():
    r = calc([item(card_discounts=[CARD_RATE])], cards=["한결카드"])[0]
    assert r.card_discount == 10000 and r.real_price == 193000
    assert r.applied[0].kind == "card" and r.applied[0].name == "한결카드"


@pytest.mark.parametrize("price, expected_discount", [
    (99999, 0),        # 최소 결제액 1원 미달 → 미적용
    (100000, 5000),    # 최소 결제액 정확히 충족 → 적용
])
def test_calc_card_min_amount_boundary(price, expected_discount):
    r = calc([item(price=price, card_discounts=[CARD_RATE])], cards=["한결카드"])[0]
    assert r.card_discount == expected_discount
    assert r.real_price == price - expected_discount + 3000
    if expected_discount == 0:
        assert any("최소 결제액" in s for s in r.not_applied)


def test_calc_card_min_amount_excludes_shipping():
    # 상품 98,000 + 배송비 3,000 = 101,000이지만 판정은 배송비 제외 금액 기준 → 미적용
    r = calc([item(price=98000, card_discounts=[CARD_RATE])], cards=["한결카드"])[0]
    assert r.card_discount == 0 and r.real_price == 101000


@pytest.mark.parametrize("price, expected_discount", [
    (299999, 14999),   # 5% = 14,999.95 → 원 단위 절사, 상한 미만
    (300000, 15000),   # 5% = 15,000 → 상한과 같음
    (500000, 15000),   # 5% = 25,000 → 상한 15,000으로 제한
])
def test_calc_card_max_discount_cap_and_floor(price, expected_discount):
    r = calc([item(price=price, card_discounts=[CARD_RATE])], cards=["한결카드"])[0]
    assert r.card_discount == expected_discount


def test_calc_card_amount_type():
    c = {"card": "누리카드", "type": "amount", "value": 20000, "min_amount": 300000}
    assert calc([item(price=299000, card_discounts=[c])], cards=["누리카드"])[0].card_discount == 0
    assert calc([item(price=300000, card_discounts=[c])], cards=["누리카드"])[0].card_discount == 20000


def test_calc_only_best_single_card():
    cards = [CARD_RATE, {"card": "누리카드", "type": "amount", "value": 12000, "min_amount": 100000}]
    r = calc([item(price=200000, card_discounts=cards)], cards=["한결카드", "누리카드"])[0]
    # 한결 10,000 vs 누리 12,000 → 누리 1장만
    assert r.card_discount == 12000 and [a.name for a in r.applied] == ["누리카드"]


def test_calc_membership_free_shipping():
    m = {"membership": "바로플러스", "type": "rate", "value": 0.0, "max_discount": 0, "free_shipping": True}
    r = calc([item(membership_discounts=[m])], memberships=["바로플러스"])[0]
    assert r.shipping_paid == 0 and r.real_price == 200000
    assert [a.kind for a in r.applied] == ["free_shipping"]
    # 멤버십 미보유면 배송비 그대로
    assert calc([item(membership_discounts=[m])])[0].real_price == 203000


def test_calc_membership_then_card_order():
    # 멤버십 5%(10,000) 먼저 → 190,000에 카드 5% = 9,500, 무료배송
    m = {"membership": "바로플러스", "type": "rate", "value": 0.05, "max_discount": 30000, "free_shipping": True}
    r = calc([item(card_discounts=[CARD_RATE], membership_discounts=[m])],
             cards=["한결카드"], memberships=["바로플러스"])[0]
    assert r.membership_discount == 10000
    assert r.card_discount == 9500
    assert r.shipping_paid == 0
    assert r.real_price == 200000 - 10000 - 9500
    assert [a.kind for a in r.applied] == ["membership", "free_shipping", "card"]


def test_calc_membership_lowers_base_below_card_min():
    # 멤버십 적용 후 상품 금액이 카드 최소 결제액 미만이 되면 카드 미적용
    m = {"membership": "누리패스", "type": "amount", "value": 5000, "min_amount": 100000, "free_shipping": False}
    r = calc([item(price=103000, card_discounts=[CARD_RATE], membership_discounts=[m])],
             cards=["한결카드"], memberships=["누리패스"])[0]
    assert r.membership_discount == 5000 and r.card_discount == 0
    assert r.real_price == 98000 + 3000


def test_calc_membership_amount_min_amount_and_free_shipping_kept():
    # 멤버십 정액 할인은 최소 금액 미달로 미적용, 무료배송은 유지
    m = {"membership": "누리패스", "type": "amount", "value": 5000, "min_amount": 100000, "free_shipping": True}
    r = calc([item(price=90000, membership_discounts=[m])], memberships=["누리패스"])[0]
    assert r.membership_discount == 0 and r.shipping_paid == 0 and r.real_price == 90000


def test_calc_batch_keeps_input_order_and_tool_invoke():
    products = search_products.invoke({"keyword": "클린웨이브 V7"})["products"]
    vc_m1 = [p for p in products if p["model_id"] == "VC-M1"]
    out = calc_real_price.invoke({"items": vc_m1, "user_cards": ["한결카드"],
                                  "user_memberships": ["바로플러스"]})
    ids = [r["product_id"] for r in out["results"]]
    assert ids == [p["product_id"] for p in vc_m1]
    assert all(r["real_price"] > 0 for r in out["results"])
    assert "샘플 데이터" in out["notice"]


def test_calc_empty_user_lists_default():
    out = calc_real_price.invoke({"items": [item()]})
    assert out["results"][0]["real_price"] == 203000


def test_search_products_drops_weak_matches_for_multi_token_keyword():
    out = search_products.invoke({"keyword": "클린웨이브 V7 사일런트 무선청소기"})
    assert out["model_ids"] == ["VC-M1"]          # 같은 브랜드 다른 카테고리·다른 무선청소기 제외
    assert {p["category"] for p in out["products"]} == {"무선청소기"}
