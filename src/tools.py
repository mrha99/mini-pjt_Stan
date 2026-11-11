"""도메인 도구 5종, 재시도 (패턴 4)

도구 이름·입출력은 고정이다 (trace의 step·test_queries.csv의 expected_tools와 대조).

| 도구               | 입력 → 출력                                                        |
| ------------------ | ------------------------------------------------------------------ |
| search_products    | 키워드, 가격 범위 → 상품·가격·판매처 목록 (product_id·model_id·할인) |
| fetch_reviews      | 상품명 목록(+조건어) → 리뷰 원문 (sample: 모델 단위 / blog: 참고 후기) |
| retrieve_reviews   | 확장 쿼리 목록(+model_ids) → 리랭킹된 리뷰 청크 Top k (retriever.py)  |
| query_price_db     | 상품 ID 목록 → 30일 가격 이력 (SQLite)                              |
| calc_real_price    | 판매처별 상품 목록 + 보유 카드·멤버십 → 상품별 실결제가 (순수 함수)  |

- 각 도구는 LangChain `@tool`이며 입력은 Pydantic args_schema, 출력은 Pydantic 모델을
  `model_dump(mode="json")`한 dict다 (ToolMessage에는 JSON 문자열로 들어간다).
- 외부 호출(NAVER API HUB 블로그 검색)은 최대 3회 지수 백오프 재시도 후 SQLite 캐시
  (`data/prices.db`의 `api_cache` 테이블)로 대체한다. seed_db.py는 price_history·meta만
  다시 만들므로 캐시는 유지된다.
- 상품·가격·판매처·할인 규칙은 모두 샘플(가짜) 데이터다.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Literal, Protocol

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.guardrails import mask_pii

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_PATH = ROOT / "data" / "products.json"
REVIEWS_PATH = ROOT / "data" / "reviews" / "sample_reviews.json"
DB_PATH = ROOT / "data" / "prices.db"

load_dotenv(ROOT / ".env", override=False)

SAMPLE_NOTICE = "샘플 데이터(가짜): 상품·가격·판매처·할인 규칙은 과제용 가상 데이터이며 실제 시세가 아닙니다."
BLOG_LABEL = "실제 블로그 참고 후기"
BLOG_API_URL = "https://naverapihub.apigw.ntruss.com/search/v1/blog"
MAX_SEARCH_RESULTS = 20
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.5  # 초. 0.5 → 1.0 (시도 사이 대기)


# ─────────────────────────────── 공통 스키마 ───────────────────────────────


class CardDiscount(BaseModel):
    card: str
    type: Literal["rate", "amount"]
    value: float = Field(description="rate면 할인율(0.05=5%), amount면 정액(원)")
    min_amount: int = Field(default=0, description="이 금액 이상 결제 시 적용")
    max_discount: int | None = Field(default=None, description="최대 할인액(원)")


class MembershipDiscount(BaseModel):
    membership: str
    type: Literal["rate", "amount"]
    value: float = Field(description="rate면 할인율, amount면 정액(원)")
    min_amount: int = Field(default=0)
    max_discount: int | None = None
    free_shipping: bool = False


# ─────────────────────────── 상품 데이터 소스 (교체 가능) ───────────────────────────


class ProductSource(Protocol):
    """상품·가격 데이터 소스 인터페이스.

    공식 가격 API를 확보하면 이 프로토콜을 구현한 클래스로 `set_product_source()`만
    바꾸면 된다. 반환 dict는 products.json 상품 항목과 같은 키를 가진다.
    """

    def search(self, keyword: str, min_price: int | None, max_price: int | None,
               limit: int) -> list[dict]: ...

    def find_by_name(self, name: str) -> dict | None: ...

    def list_by_model(self, model_id: str) -> list[dict]: ...


_TOKEN_SPLIT = re.compile(r"[\s,·/]+")


class JsonCatalogSource:
    """data/products.json 샘플 카탈로그 소스."""

    def __init__(self, path: Path | None = None):
        self.path = path

    def _products(self) -> list[dict]:
        path = self.path or PRODUCTS_PATH
        return json.loads(path.read_text(encoding="utf-8"))["products"]

    @staticmethod
    def _haystack(p: dict) -> str:
        return " ".join([p["name"], p["brand"], p["category"], p["model_id"],
                         p["product_id"], p["mall"]]).lower()

    def search(self, keyword: str, min_price: int | None, max_price: int | None,
               limit: int) -> list[dict]:
        tokens = [t for t in _TOKEN_SPLIT.split(keyword.lower()) if t]
        scored = []
        for p in self._products():
            if min_price is not None and p["price"] < min_price:
                continue
            if max_price is not None and p["price"] > max_price:
                continue
            hay = self._haystack(p)
            score = sum(1 for t in tokens if t in hay) if tokens else 1
            if score > 0:
                scored.append((score, p))
        # 일치 토큰 수 내림차순 → 판매가 오름차순
        scored.sort(key=lambda x: (-x[0], x[1]["price"]))
        return [p for _, p in scored[:limit]]

    def find_by_name(self, name: str) -> dict | None:
        key = name.strip().lower()
        if not key:
            return None
        products = self._products()
        for p in products:  # 정확히 일치 (상품명·model_id·product_id)
            if key in (p["name"].lower(), p["model_id"].lower(), p["product_id"].lower()):
                return p
        for p in products:  # 부분 일치
            if key in p["name"].lower() or p["name"].lower() in key:
                return p
        return None

    def list_by_model(self, model_id: str) -> list[dict]:
        return [p for p in self._products() if p["model_id"] == model_id]


_product_source: ProductSource = JsonCatalogSource()


def get_product_source() -> ProductSource:
    return _product_source


def set_product_source(source: ProductSource) -> None:
    global _product_source
    _product_source = source


# ─────────────────────────────── 캐시 (SQLite) ───────────────────────────────

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key  TEXT PRIMARY KEY,   -- 요청 해시 (질의 원문은 저장하지 않음)
    source     TEXT NOT NULL,
    payload    TEXT NOT NULL,      -- JSON (PII 마스킹 후)
    fetched_at TEXT NOT NULL       -- ISO8601 UTC
);
"""

_mask_pii = mask_pii  # 캐시·출력에 남기기 전 PII 마스킹 (규칙은 guardrails.py)


def _cache_key(source: str, request: dict) -> str:
    raw = json.dumps({"source": source, **request}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_put(key: str, source: str, payload: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_CACHE_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO api_cache VALUES (?, ?, ?, ?)",
            (key, source, json.dumps(payload, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


def _cache_get(key: str) -> tuple[dict, str] | None:
    if not DB_PATH.exists():
        return None
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_CACHE_SCHEMA)
        row = conn.execute(
            "SELECT payload, fetched_at FROM api_cache WHERE cache_key = ?", (key,)
        ).fetchone()
    return (json.loads(row[0]), row[1]) if row else None


def _with_retry(fn, attempts: int = RETRY_ATTEMPTS, base_delay: float = RETRY_BASE_DELAY):
    """fn()을 최대 attempts회 시도, 실패 사이에 지수 백오프(base, 2·base, ...)."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 네트워크·HTTP 오류 모두 재시도
            last_exc = exc
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────── 1. search_products ───────────────────────────────


class SearchProductsInput(BaseModel):
    keyword: str = Field(description="검색 키워드 (예: '무선청소기', '사일런트 청소기')")
    min_price: int | None = Field(default=None, ge=0, description="최소 판매가(원)")
    max_price: int | None = Field(default=None, ge=0, description="최대 판매가(원), 예산 상한")
    limit: int = Field(default=MAX_SEARCH_RESULTS, ge=1, le=MAX_SEARCH_RESULTS)


class Product(BaseModel):
    product_id: str = Field(description="판매처별 상품 ID")
    model_id: str = Field(description="같은 모델 묶음 ID (카드 묶기·model_ids 전달용)")
    name: str
    brand: str
    category: str
    mall: str = Field(description="판매처")
    price: int = Field(description="판매가(원)")
    shipping_fee: int = Field(description="배송비(원), 0이면 무료배송")
    card_discounts: list[CardDiscount]
    membership_discounts: list[MembershipDiscount]
    link: str
    specs: dict = Field(default_factory=dict)
    is_sample: bool = True


class SearchProductsOutput(BaseModel):
    products: list[Product]
    count: int
    model_ids: list[str] = Field(description="결과에 포함된 서로 다른 model_id (등장 순)")
    notice: str = SAMPLE_NOTICE


def search_products_impl(keyword: str, min_price: int | None = None,
                         max_price: int | None = None,
                         limit: int = MAX_SEARCH_RESULTS) -> SearchProductsOutput:
    raw = get_product_source().search(keyword, min_price, max_price, min(limit, MAX_SEARCH_RESULTS))
    products = [Product.model_validate(p) for p in raw]
    model_ids = list(dict.fromkeys(p.model_id for p in products))
    return SearchProductsOutput(products=products, count=len(products), model_ids=model_ids)


@tool("search_products", args_schema=SearchProductsInput)
def search_products(keyword: str, min_price: int | None = None, max_price: int | None = None,
                    limit: int = MAX_SEARCH_RESULTS) -> dict:
    """키워드와 가격 범위(판매가 기준)로 여러 판매처의 상품 후보를 최대 20개 찾는다.
    각 상품에 product_id, model_id, 판매처, 가격, 배송비, 카드·멤버십 할인 정보, 링크가 있다.
    결과가 비면 상품을 찾지 못한 것이다(가격을 지어내지 말 것)."""
    return search_products_impl(keyword, min_price, max_price, limit).model_dump(mode="json")


# ─────────────────────────────── 2. fetch_reviews ───────────────────────────────


class FetchReviewsInput(BaseModel):
    product_names: list[str] = Field(min_length=1, description="후보 상품명 목록 (여러 건 한 번에)")
    conditions: list[str] = Field(default_factory=list,
                                  description="선택: 사용자 조건어 (예: ['원룸', '소음'])")


class Review(BaseModel):
    review_id: str = Field(description="sample: rv-XXXX / blog: blog-{글번호}")
    source: Literal["sample", "blog"]
    label: str = Field(description="표시 라벨 (blog면 '실제 블로그 참고 후기')")
    model_id: str | None = Field(default=None, description="blog 결과는 특정 상품 근거가 아니므로 None")
    product_id: str | None = None
    mall: str | None = None
    category: str | None = None
    title: str
    body: str
    author: str | None = None
    date: str | None = None
    rating: int | None = None
    link: str


class FetchReviewsOutput(BaseModel):
    source: Literal["sample", "blog"]
    reviews: list[Review]
    count: int
    model_ids: list[str] = Field(default_factory=list, description="sample: 리뷰를 모은 모델")
    not_found: list[str] = Field(default_factory=list, description="카탈로그에서 못 찾은 상품명")
    no_review_models: list[str] = Field(default_factory=list, description="리뷰가 하나도 없는 모델")
    indexed: int = Field(default=0, description="벡터 DB 색인에 들어 있는 리뷰 수 (새로 추가 + 기존)")
    from_cache: bool = False
    cached_at: str | None = None
    notice: str = ""


def _review_source() -> str:
    src = os.getenv("REVIEW_SOURCE", "sample").strip().lower()
    return src if src in ("sample", "blog") else "sample"


def _index_reviews(reviews: list[Review]) -> int:
    """indexer.index_reviews로 벡터 DB·BM25에 추가한다 (기존 ID는 건너뜀, blog는 마스킹 후 저장)."""
    from src import indexer

    return int(indexer.index_reviews([r.model_dump(mode="json") for r in reviews]))


def _fetch_sample(product_names: list[str]) -> FetchReviewsOutput:
    source = get_product_source()
    all_reviews = json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]

    model_ids: list[str] = []
    not_found: list[str] = []
    for name in product_names:
        p = source.find_by_name(name)
        if p is None:
            not_found.append(name)
        elif p["model_id"] not in model_ids:
            model_ids.append(p["model_id"])

    reviews: list[Review] = []
    no_review_models: list[str] = []
    for mid in model_ids:
        listings = {p["product_id"]: p for p in source.list_by_model(mid)}  # 모델의 모든 판매처
        found = [r for r in all_reviews if r["product_id"] in listings]
        if not found:
            no_review_models.append(mid)
        for r in found:
            p = listings[r["product_id"]]
            reviews.append(Review(
                review_id=r["review_id"], source="sample", label="샘플 리뷰(가짜)",
                model_id=mid, product_id=r["product_id"], mall=p["mall"], category=p["category"],
                title=r["title"], body=r["body"], author=r.get("author"), date=r.get("date"),
                rating=r.get("rating"), link=r["link"],
            ))
    return FetchReviewsOutput(
        source="sample", reviews=reviews, count=len(reviews), model_ids=model_ids,
        not_found=not_found, no_review_models=no_review_models,
        notice="샘플 리뷰(가짜): 모델 단위로 모든 판매처 리뷰를 모았습니다.",
    )


_TAG_RE = re.compile(r"<[^>]+>")
_BLOG_NO_RE = re.compile(r"(\d{6,})(?:\D*)$")


def clean_snippet(text: str) -> str:
    """블로그 API 스니펫의 HTML 태그(<b> 등) 제거 후 엔티티(&quot; 등) 해제."""
    text = _TAG_RE.sub("", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _blog_review_id(link: str) -> str:
    m = _BLOG_NO_RE.search(link or "")
    if m:
        return f"blog-{m.group(1)}"
    return "blog-" + hashlib.sha1((link or "").encode("utf-8")).hexdigest()[:10]


def _http_client() -> httpx.Client:
    """테스트에서 MockTransport로 교체할 수 있도록 분리."""
    return httpx.Client(timeout=10.0)


def _call_blog_api(query: str, display: int = 10) -> dict:
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("NAVER API 키가 설정되지 않았습니다")
    headers = {"X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret}
    params = {"query": query, "display": display, "sort": "sim"}
    with _http_client() as client:
        resp = client.get(BLOG_API_URL, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()


def _parse_blog_items(data: dict, category: str) -> list[dict]:
    items = []
    for it in data.get("items", []):
        link = it.get("link", "")
        items.append({
            "review_id": _blog_review_id(link),
            "category": category,
            "title": _mask_pii(clean_snippet(it.get("title", ""))),
            "body": _mask_pii(clean_snippet(it.get("description", ""))),
            "author": clean_snippet(it.get("bloggername", "")) or None,
            "date": it.get("postdate") or None,
            "link": link,
        })
    return items


def _fetch_blog(product_names: list[str], conditions: list[str]) -> FetchReviewsOutput:
    source = get_product_source()
    categories: list[str] = []
    not_found: list[str] = []
    for name in product_names:
        p = source.find_by_name(name)
        if p is None:
            not_found.append(name)
        elif p["category"] not in categories:
            categories.append(p["category"])

    reviews: list[Review] = []
    used_cache = False
    cache_times: list[str] = []
    failed: list[str] = []
    seen: set[str] = set()
    for category in categories:
        # 상품명이 가상이라 "카테고리 + 조건어 + 후기"로 검색한다
        query = " ".join([category, *conditions, "후기"])
        key = _cache_key("naver_blog", {"query": query})
        try:
            data = _with_retry(lambda q=query: _call_blog_api(q))
            items = _parse_blog_items(data, category)
            _cache_put(key, "naver_blog", {"items": items})
        except Exception:  # noqa: BLE001 - 재시도 실패 시 캐시로 대체
            cached = _cache_get(key)
            if cached is None:
                failed.append(category)
                continue
            items = cached[0]["items"]
            used_cache = True
            cache_times.append(cached[1])
        for it in items:
            if it["review_id"] in seen:
                continue
            seen.add(it["review_id"])
            reviews.append(Review(source="blog", label=BLOG_LABEL, model_id=None, **it))

    notes = [f"{BLOG_LABEL}: 카테고리+조건어로 검색한 실제 블로그 스니펫이며 특정 상품의 근거가 아닙니다."]
    cached_at = min(cache_times) if cache_times else None
    if used_cache:
        notes.append(f"블로그 API 호출 실패로 캐시된 결과를 사용했습니다 (기준 시점 {cached_at}).")
    if failed:
        notes.append(f"블로그 API 호출 실패, 캐시 없음: {', '.join(failed)}")
    return FetchReviewsOutput(
        source="blog", reviews=reviews, count=len(reviews), not_found=not_found,
        from_cache=used_cache, cached_at=cached_at, notice=" ".join(notes),
    )


def fetch_reviews_impl(product_names: list[str],
                       conditions: list[str] | None = None) -> FetchReviewsOutput:
    conditions = conditions or []
    if _review_source() == "blog":
        out = _fetch_blog(product_names, conditions)
    else:
        out = _fetch_sample(product_names)
    out.indexed = _index_reviews(out.reviews)
    return out


@tool("fetch_reviews", args_schema=FetchReviewsInput)
def fetch_reviews(product_names: list[str], conditions: list[str] | None = None) -> dict:
    """후보 상품명 목록의 리뷰를 한 번에 수집해 벡터 DB에 색인한다.
    sample 모드는 모델(model_id) 단위로 모든 판매처의 리뷰를 모으고,
    blog 모드는 '카테고리 + 조건어 + 후기'로 검색한 '실제 블로그 참고 후기'를 돌려준다(상품 근거 아님)."""
    return fetch_reviews_impl(product_names, conditions).model_dump(mode="json")


# ─────────────────────────────── 3. retrieve_reviews ───────────────────────────────


class RetrieveReviewsInput(BaseModel):
    queries: list[str] = Field(min_length=1, description="조건어로 확장한 검색 쿼리 목록")
    model_ids: list[str] | None = Field(default=None,
                                        description="선택: 후보 model_id 목록. 있으면 이 모델 리뷰만 검색")
    # 반환 청크 수 k는 입력으로 받지 않고 설정값 RETRIEVER_TOP_K(기본 8)로 고정한다 (1·2차 평가 공정 비교).


class ReviewChunk(BaseModel):
    doc_id: str = Field(description="{리뷰ID}#c{번호}")
    text: str
    review_id: str | None = None
    model_id: str | None = None
    product_id: str | None = None
    mall: str | None = None
    category: str | None = None
    score: float | None = None
    source: str | None = None
    title: str | None = None
    link: str | None = None


class RetrieveReviewsOutput(BaseModel):
    chunks: list[ReviewChunk] = Field(description="답변 근거 청크 (API contexts로 사용)")
    count: int
    blog_references: list[ReviewChunk] = Field(
        default_factory=list, description="'실제 블로그 참고 후기' (근거 아님, 카드 아래 별도 표시)")
    status: Literal["ok", "empty_index", "error"] = "ok"
    notice: str = ""


def retrieve_reviews_impl(queries: list[str],
                          model_ids: list[str] | None = None) -> RetrieveReviewsOutput:
    """retriever.retrieve(현재 RETRIEVER_MODE) + retrieve_blog_refs를 호출한다."""
    from src import indexer, retriever

    try:
        if indexer.get_collection().count() == 0:
            return RetrieveReviewsOutput(chunks=[], count=0, status="empty_index",
                                         notice="리뷰 색인이 비어 있습니다 (python -m src.indexer로 색인).")
        chunks = [ReviewChunk.model_validate(c) for c in retriever.retrieve(queries, model_ids)]
        refs = [ReviewChunk.model_validate(c) for c in retriever.retrieve_blog_refs(queries, model_ids)]
    except Exception as exc:  # noqa: BLE001 - 에이전트가 오류를 보고 다음 행동을 고를 수 있게
        return RetrieveReviewsOutput(chunks=[], count=0, status="error",
                                     notice=f"리뷰 검색 실패: {type(exc).__name__}")
    notice = "" if chunks else "조건에 맞는 리뷰 근거를 찾지 못했습니다 (근거 없음 → '확인 불가')."
    if refs:
        notice = (notice + f" {BLOG_LABEL} {len(refs)}건은 특정 상품의 근거가 아닙니다.").strip()
    return RetrieveReviewsOutput(chunks=chunks, count=len(chunks), blog_references=refs, notice=notice)


@tool("retrieve_reviews", args_schema=RetrieveReviewsInput)
def retrieve_reviews(queries: list[str], model_ids: list[str] | None = None) -> dict:
    """확장 쿼리 목록으로 색인된 리뷰를 검색해 상위 청크 8개를 돌려준다.
    queries[0]에는 사용자 질의(핵심어+조건), 나머지에는 조건어(예: '원룸', '소음')를 넣는다.
    추천 요청에서는 search_products가 찾은 후보 model_ids를 넘겨 검색 범위를 제한한다.
    blog_references는 '실제 블로그 참고 후기'로 답변 근거로 쓰지 않는다."""
    return retrieve_reviews_impl(queries, model_ids).model_dump(mode="json")


# ─────────────────────────────── 4. query_price_db ───────────────────────────────


class QueryPriceDbInput(BaseModel):
    product_ids: list[str] = Field(min_length=1, description="상품 ID 목록 (예: ['VC-001'])")
    days: int = Field(default=30, ge=1, le=30, description="최근 며칠치 (최대 30)")


class PricePoint(BaseModel):
    date: str
    price: int
    shipping_fee: int


class PriceHistory(BaseModel):
    product_id: str
    model_id: str
    mall: str
    points: list[PricePoint]
    current_price: int = Field(description="기준일(마지막 날) 판매가")
    min_price: int
    min_date: str
    max_price: int
    avg_price: int = Field(description="기간 평균 판매가 (원 단위 절사)")


class QueryPriceDbOutput(BaseModel):
    histories: list[PriceHistory]
    not_found: list[str] = Field(default_factory=list)
    period: str = ""
    notice: str = "샘플 데이터(가짜): 샘플 카탈로그로 생성한 30일 가격 이력이며 참고용입니다."


def query_price_db_impl(product_ids: list[str], days: int = 30) -> QueryPriceDbOutput:
    if not DB_PATH.exists():
        return QueryPriceDbOutput(histories=[], not_found=list(product_ids),
                                  notice="가격 이력 DB가 없습니다 (python -m src.seed_db로 생성).")
    histories: list[PriceHistory] = []
    not_found: list[str] = []
    with sqlite3.connect(DB_PATH) as conn:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        for pid in dict.fromkeys(product_ids):
            rows = conn.execute(
                "SELECT model_id, mall, date, price, shipping_fee FROM price_history "
                "WHERE product_id = ? ORDER BY date DESC LIMIT ?", (pid, days),
            ).fetchall()
            if not rows:
                not_found.append(pid)
                continue
            rows.reverse()
            points = [PricePoint(date=r[2], price=r[3], shipping_fee=r[4]) for r in rows]
            low = min(points, key=lambda pt: (pt.price, pt.date))
            histories.append(PriceHistory(
                product_id=pid, model_id=rows[0][0], mall=rows[0][1], points=points,
                current_price=points[-1].price, min_price=low.price, min_date=low.date,
                max_price=max(pt.price for pt in points),
                avg_price=sum(pt.price for pt in points) // len(points),
            ))
    period = f"{meta.get('start_date', '?')} ~ {meta.get('end_date', '?')}"
    return QueryPriceDbOutput(histories=histories, not_found=not_found, period=period)


@tool("query_price_db", args_schema=QueryPriceDbInput)
def query_price_db(product_ids: list[str], days: int = 30) -> dict:
    """상품 ID 목록의 최근 가격 이력(최대 30일)과 최저·최고·평균·현재가를 조회한다 (샘플 데이터)."""
    return query_price_db_impl(product_ids, days).model_dump(mode="json")


# ─────────────────────────────── 5. calc_real_price ───────────────────────────────


class PriceItem(BaseModel):
    product_id: str
    model_id: str | None = None
    mall: str | None = None
    price: int = Field(ge=0, description="판매가(원)")
    shipping_fee: int = Field(default=0, ge=0)
    card_discounts: list[CardDiscount] = Field(default_factory=list)
    membership_discounts: list[MembershipDiscount] = Field(default_factory=list)


class CalcRealPriceInput(BaseModel):
    items: list[PriceItem] = Field(min_length=1, description="판매처별 상품 목록 (여러 건 한 번에)")
    user_cards: list[str] = Field(default_factory=list, description="사용자가 가진 카드 (없으면 빈 목록)")
    user_memberships: list[str] = Field(default_factory=list, description="사용자가 가진 멤버십")


class AppliedDiscount(BaseModel):
    kind: Literal["membership", "card", "free_shipping"]
    name: str
    amount: int = Field(description="절감액(원)")
    detail: str


class RealPrice(BaseModel):
    product_id: str
    model_id: str | None = None
    mall: str | None = None
    price: int
    shipping_fee: int = Field(description="원래 배송비")
    membership_discount: int = 0
    card_discount: int = 0
    shipping_paid: int = Field(description="실제 부담 배송비")
    real_price: int = Field(description="실결제가 = 상품 금액 - 할인 + 배송비")
    applied: list[AppliedDiscount] = Field(default_factory=list, description="적용된 할인 내역")
    not_applied: list[str] = Field(default_factory=list, description="보유했지만 적용되지 않은 할인과 사유")


class CalcRealPriceOutput(BaseModel):
    results: list[RealPrice]
    notice: str = "샘플 데이터(가짜) 할인 규칙 기준 계산이며 원 단위 미만은 절사했습니다."


def _floor_won(amount: Decimal) -> int:
    return int(amount.to_integral_value(rounding=ROUND_FLOOR))


def _discount_amount(base: int, dtype: str, value: float, min_amount: int,
                     max_discount: int | None) -> tuple[int, str | None]:
    """(할인액, 미적용 사유). base는 배송비를 뺀 상품 금액."""
    if base < min_amount:
        return 0, f"최소 결제액 {min_amount:,}원 미달 (상품 금액 {base:,}원)"
    if dtype == "rate":
        amt = _floor_won(Decimal(base) * Decimal(str(value)))
    else:
        amt = _floor_won(Decimal(str(value)))
    if max_discount is not None:
        amt = min(amt, max_discount)
    return min(amt, base), None


def _fmt_rule(dtype: str, value: float, max_discount: int | None) -> str:
    rule = f"{(Decimal(str(value)) * 100).normalize():f}%" if dtype == "rate" else f"{int(value):,}원"
    if max_discount is not None:
        rule += f" (최대 {max_discount:,}원)"
    return rule


def _calc_one(item: PriceItem, cards: set[str], memberships: set[str]) -> RealPrice:
    applied: list[AppliedDiscount] = []
    not_applied: list[str] = []

    # 1) 멤버십: 보유한 것 중 (할인액 + 면제 배송비)가 가장 큰 1개
    best_m: tuple[int, int, bool, MembershipDiscount] | None = None
    for m in item.membership_discounts:
        if m.membership not in memberships:
            continue
        amt, reason = _discount_amount(item.price, m.type, m.value, m.min_amount, m.max_discount)
        if reason:
            not_applied.append(f"{m.membership} 할인: {reason}")
        saved = amt + (item.shipping_fee if m.free_shipping else 0)
        if best_m is None or saved > best_m[0]:
            best_m = (saved, amt, m.free_shipping, m)

    membership_discount = 0
    shipping_paid = item.shipping_fee
    if best_m is not None:
        _, amt, free_ship, m = best_m
        membership_discount = amt
        if amt > 0:
            applied.append(AppliedDiscount(kind="membership", name=m.membership, amount=amt,
                                           detail=_fmt_rule(m.type, m.value, m.max_discount)))
        if free_ship:
            if item.shipping_fee > 0:
                applied.append(AppliedDiscount(kind="free_shipping", name=m.membership,
                                               amount=item.shipping_fee, detail="멤버십 무료배송"))
            shipping_paid = 0

    # 2) 카드: 멤버십 적용 후 상품 금액(배송비 제외) 기준, 할인액이 가장 큰 1장
    base = item.price - membership_discount
    best_c: tuple[int, CardDiscount] | None = None
    for c in item.card_discounts:
        if c.card not in cards:
            continue
        amt, reason = _discount_amount(base, c.type, c.value, c.min_amount, c.max_discount)
        if reason:
            not_applied.append(f"{c.card}: {reason}")
            continue
        if best_c is None or amt > best_c[0]:
            best_c = (amt, c)
    card_discount = 0
    if best_c is not None and best_c[0] > 0:
        card_discount, c = best_c
        applied.append(AppliedDiscount(kind="card", name=c.card, amount=card_discount,
                                       detail=_fmt_rule(c.type, c.value, c.max_discount)))

    real_price = base - card_discount + shipping_paid
    return RealPrice(
        product_id=item.product_id, model_id=item.model_id, mall=item.mall,
        price=item.price, shipping_fee=item.shipping_fee,
        membership_discount=membership_discount, card_discount=card_discount,
        shipping_paid=shipping_paid, real_price=real_price,
        applied=applied, not_applied=not_applied,
    )


def calc_real_price_impl(items: list[PriceItem | dict], user_cards: list[str] | None = None,
                         user_memberships: list[str] | None = None) -> CalcRealPriceOutput:
    """순수 함수: 입력 순서대로 상품별 실결제가를 계산한다 (I/O 없음)."""
    cards = set(user_cards or [])
    memberships = set(user_memberships or [])
    parsed = [i if isinstance(i, PriceItem) else PriceItem.model_validate(i) for i in items]
    return CalcRealPriceOutput(results=[_calc_one(i, cards, memberships) for i in parsed])


@tool("calc_real_price", args_schema=CalcRealPriceInput)
def calc_real_price(items: list[PriceItem], user_cards: list[str] | None = None,
                    user_memberships: list[str] | None = None) -> dict:
    """판매처별 상품 목록(판매가·배송비·카드/멤버십 할인)과 사용자 보유 카드·멤버십으로
    상품별 실결제가와 적용 내역을 한 번에 계산한다. 멤버십 → 카드(가장 큰 1장) 순서로 적용한다."""
    return calc_real_price_impl(items, user_cards, user_memberships).model_dump(mode="json")


ALL_TOOLS = [search_products, fetch_reviews, retrieve_reviews, query_price_db, calc_real_price]
