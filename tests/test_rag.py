"""RAG 파이프라인 테스트 (indexer·retriever). conftest의 가짜 임베딩·리랭커와 임시 Chroma를 쓴다."""

from __future__ import annotations

import json

import pytest

from src import indexer, retriever
from src.guardrails import mask_pii
from src.tools import BLOG_LABEL, fetch_reviews, retrieve_reviews

QUERY = "원룸에서 쓸 35만원 이하 무선청소기 추천해줘. 소음이 제일 중요해"
BLOG_REVIEWS = [
    {"review_id": "blog-223456789", "source": "blog", "category": "무선청소기",
     "title": "원룸 무선청소기 소음 후기", "link": "https://blog.naver.com/a/223456789",
     "body": "원룸에서 밤에 돌려도 조용해요. 문의 010-1234-5678, a.b@example.com, "
             "서울시 강남구 테헤란로 123, 계좌 110-123-456789"},
    {"review_id": "blog-223000111", "source": "blog", "category": "공기청정기",
     "title": "공기청정기 후기", "link": "https://blog.naver.com/b/223000111",
     "body": "미세먼지 수치가 확실히 내려가요. 수면모드가 조용해요."},
]


@pytest.fixture
def built():
    return indexer.build_index()


def _sample_reviews() -> list[dict]:
    return json.loads(indexer.SAMPLE_REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]


# ─────────────────────────────── indexer ───────────────────────────────


def test_chunk_ids_and_metadata_from_catalog():
    review = next(r for r in _sample_reviews() if r["review_id"] == "rv-0001")
    chunks = indexer.review_to_chunks(review)
    assert [c[0] for c in chunks] == [f"rv-0001#c{i}" for i in range(len(chunks))]
    meta = chunks[0][2]
    # 리뷰 JSON에 없는 model_id·category를 products.json에서 채운다
    assert meta["review_id"] == "rv-0001" and meta["product_id"] == "VC-001"
    assert meta["model_id"] == "VC-M1" and meta["category"] == "무선청소기"
    assert meta["source"] == "sample"
    assert all(len(text) <= indexer.CHUNK_SIZE for _, text, _ in chunks)


def test_blog_chunk_has_no_product_link():
    meta = indexer.review_to_chunks(BLOG_REVIEWS[0])[0][2]
    assert meta["source"] == "blog" and meta["category"] == "무선청소기"
    assert "model_id" not in meta and "product_id" not in meta


def test_build_index_all_samples_and_idempotent(built):
    n_reviews = len(_sample_reviews())
    assert built["added"] == n_reviews and built["total_chunks"] >= n_reviews
    again = indexer.build_index()
    assert again["added"] == 0 and again["skipped"] == n_reviews
    assert again["total_chunks"] == built["total_chunks"]


def test_bm25_matches_chroma_and_updates_on_add(built):
    bm25 = indexer.get_bm25()
    col = indexer.get_collection()
    assert len(bm25) == col.count() and set(bm25.ids) == set(col.get()["ids"])
    indexer.index_reviews(BLOG_REVIEWS)
    bm25_after = indexer.get_bm25()
    assert len(bm25_after) == col.count() > len(bm25)
    assert "blog-223456789#c0" in bm25_after.ids


def test_index_reviews_blog_masks_saves_and_skips_existing(built):
    assert indexer.index_reviews(BLOG_REVIEWS) == 2
    saved = json.loads(indexer.BLOG_REVIEWS_PATH.read_text(encoding="utf-8"))
    body = saved["reviews"][0]["body"]
    for raw in ("010-1234-5678", "a.b@example.com", "테헤란로 123", "110-123-456789"):
        assert raw not in body
    for tag in ("[전화번호]", "[이메일]", "[주소]", "[계좌번호]"):
        assert tag in body
    # 색인된 본문도 마스킹 상태
    docs = indexer.get_collection().get(where={"source": "blog"})["documents"]
    assert all("010-1234-5678" not in d for d in docs)

    count = indexer.get_collection().count()
    assert indexer.index_reviews(BLOG_REVIEWS) == 2  # 이미 있음 → 건너뜀, 색인에는 있음
    assert indexer.get_collection().count() == count
    assert len(json.loads(indexer.BLOG_REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]) == 2


def test_blog_review_id_from_link_last_number():
    r = {**BLOG_REVIEWS[1], "review_id": "tmp", "source": "blog",
         "link": "https://blog.naver.com/x/224999000"}
    indexer.index_reviews([r])
    ids = indexer.get_collection().get(where={"source": "blog"})["ids"]
    assert ids == ["blog-224999000#c0"]


def test_index_mismatched_embedding_model_requires_rebuild(built):
    class Other(type(indexer.get_embedder())):
        name = "other-model"

    indexer.set_embedder(Other())
    with pytest.raises(RuntimeError, match="--rebuild"):
        indexer.get_collection()
    stats = indexer.build_index(rebuild=True)
    assert stats["added"] == len(_sample_reviews())


def test_mask_pii_patterns():
    text = "연락 010-0000-1234 / 02-123-4567, 메일 seller.help@example.com, 부산광역시 해운대구 센텀로 45 101동 202호, 국민 123456-78-901234"
    masked = mask_pii(text)
    assert masked.count("[전화번호]") == 2 and "[이메일]" in masked
    assert "[주소]" in masked and "센텀로" not in masked
    assert "[계좌번호]" in masked and "123456-78-901234" not in masked
    assert mask_pii("가격 249,000원, 무게 1.4kg, 2026-11-11") == "가격 249,000원, 무게 1.4kg, 2026-11-11"


# ─────────────────────────────── retriever: 쿼리 확장 ───────────────────────────────


def test_expand_queries_three_with_conditions():
    qs = retriever.expand_queries([QUERY])
    assert len(qs) == 3 and qs[0] == QUERY
    assert "원룸" in qs[1] and "소음" in qs[1] and "35만원" not in qs[1]
    assert "  " not in qs[1]
    assert qs[2].startswith("무선청소기") and "조용" in qs[2]


def test_conditions_ignore_category_substrings():
    # "무선청소기"의 '청소'가 '세척' 조건어로 잡히지 않아야 한다
    assert retriever.extract_conditions(["무선청소기 추천"]) == []
    assert retriever.extract_conditions(["무선청소기", "원룸", "반려동물 털"]) == ["원룸", "반려동물"]


def test_extra_query_items_become_conditions():
    conds = retriever.extract_conditions(["전기포트 추천", "손잡이 그립감"])
    assert "손잡이 그립감" in conds


# ─────────────────────────────── retriever: 검색 ───────────────────────────────


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_both_modes_return_k8_sample_only(built, mode):
    indexer.index_reviews(BLOG_REVIEWS)
    r = retriever.search([QUERY], mode=mode)
    assert len(r.chunks) == 8
    assert all(c["source"] == "sample" for c in r.chunks)  # 블로그는 근거에서 제외
    assert all("#c" in c["doc_id"] for c in r.chunks)


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_model_ids_filter_both_modes(built, mode):
    ids = ["VC-M1", "VC-M3"]
    r = retriever.search([QUERY, "원룸", "소음"], model_ids=ids, mode=mode)
    assert r.chunks and {c["model_id"] for c in r.chunks} <= set(ids)


def _models_with_reviews(ids):
    have = {m["model_id"] for m in indexer.get_collection().get()["metadatas"] if "model_id" in m}
    return [m for m in ids if m in have]


def test_hybrid_model_quota_covers_each_candidate(built):
    ids = ["VC-M1", "VC-M3", "VC-M4"]
    r = retriever.search([QUERY], model_ids=ids, mode="hybrid")
    assert len(r.chunks) == 8
    counts = {m: sum(c["model_id"] == m for c in r.chunks) for m in ids}
    quota = min(retriever.MODEL_QUOTA_MAX, 8 // len(ids))  # = 2
    for m in _models_with_reviews(ids):
        assert counts[m] >= min(quota, sum(
            1 for x in indexer.get_collection().get(where={"model_id": m})["ids"]))
    scores = [c["score"] for c in r.chunks]
    assert scores == sorted(scores, reverse=True)


def test_quota_select_rules():
    model = {"a1": "A", "a2": "A", "a3": "A", "a4": "A", "b1": "B", "c1": "C"}.get
    ordered = ["a1", "a2", "a3", "a4", "b1", "c1"]
    # 모델별 2개 먼저(B·C는 1개뿐) → 나머지는 순서대로
    assert retriever._quota_select(ordered, model, ["A", "B", "C"], 2, 5) == ["a1", "a2", "b1", "c1", "a3"]
    assert retriever._quota_select(ordered, model, ["A", "B"], 0, 3) == ["a1", "a2", "a3"]


def test_single_model_has_no_quota_effect(built):
    r = retriever.search([QUERY], model_ids=["VC-M1"], mode="hybrid")
    assert len(r.chunks) == 8 and {c["model_id"] for c in r.chunks} == {"VC-M1"}


def test_vector_mode_keeps_pure_score_order(built):
    r = retriever.search([QUERY], model_ids=["VC-M1", "VC-M3", "VC-M4"], mode="vector")
    scores = [c["score"] for c in r.chunks]
    assert scores == sorted(scores, reverse=True)


def test_empty_model_ids_returns_nothing(built):
    assert retriever.search([QUERY], model_ids=[], mode="hybrid").chunks == []


def test_vector_mode_uses_single_query_without_rerank(built, isolated_index):
    r = retriever.search([QUERY, "원룸", "소음"], mode="vector")
    assert r.queries_used == [QUERY] and r.rerank_query is None
    assert isolated_index.calls == []  # 리랭커 미호출


def test_hybrid_mode_expands_and_reranks(built, isolated_index):
    r = retriever.search([QUERY], mode="hybrid")
    assert len(r.queries_used) == 3
    assert len(isolated_index.calls) == 1
    rq, n_candidates = isolated_index.calls[0]
    assert rq == r.rerank_query and n_candidates <= retriever.RERANK_POOL
    scores = [c["score"] for c in r.chunks]
    assert scores == sorted(scores, reverse=True)
    # 모델 지정 없이도 질의의 카테고리로 범위를 좁힌다
    assert {c["category"] for c in r.chunks} == {"무선청소기"}


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_top_k_setting(built, monkeypatch, mode):
    monkeypatch.setenv("RETRIEVER_TOP_K", "5")
    assert len(retriever.search([QUERY], mode=mode).chunks) == 5


def test_mode_from_setting(built, monkeypatch):
    monkeypatch.setenv("RETRIEVER_MODE", "vector")
    assert retriever.search([QUERY]).mode == "vector"
    monkeypatch.setenv("RETRIEVER_MODE", "hybrid")
    assert retriever.search([QUERY]).mode == "hybrid"


def test_blog_refs_separate_and_category_scoped(built):
    assert retriever.retrieve_blog_refs([QUERY]) == []  # 블로그 색인 전
    indexer.index_reviews(BLOG_REVIEWS)
    refs = retriever.retrieve_blog_refs([QUERY], model_ids=["VC-M1"])
    assert [c["review_id"] for c in refs] == ["blog-223456789"]
    assert all(c["source"] == "blog" and c["model_id"] is None for c in refs)


# ─────────────────────────────── 도구 연결 ───────────────────────────────


def test_retrieve_reviews_tool_end_to_end(built, monkeypatch):
    monkeypatch.setenv("RETRIEVER_MODE", "hybrid")
    indexer.index_reviews(BLOG_REVIEWS)
    out = retrieve_reviews.invoke({"queries": [QUERY, "원룸", "소음"], "model_ids": ["VC-M1", "VC-M4"]})
    assert out["status"] == "ok" and out["count"] == 8
    assert {c["model_id"] for c in out["chunks"]} <= {"VC-M1", "VC-M4"}
    assert [c["source"] for c in out["blog_references"]] == ["blog"]
    assert BLOG_LABEL in out["notice"]


def test_fetch_reviews_sample_indexes_into_chroma(monkeypatch):
    monkeypatch.setenv("REVIEW_SOURCE", "sample")
    out = fetch_reviews.invoke({"product_names": ["클린웨이브 V7 사일런트 무선청소기"]})
    assert out["indexed"] == out["count"] > 0
    ids = {m["review_id"] for m in indexer.get_collection().get()["metadatas"]}
    assert ids == {r["review_id"] for r in out["reviews"]}


def test_compare_cli_prints_both_modes(built, capsys):
    retriever.compare(QUERY, ["VC-M1", "VC-M3"])
    out = capsys.readouterr().out
    assert "━━ vector" in out and "━━ hybrid" in out and "비교: 공통" in out


# ─────────────────────────────── 협찬 청크 제외 (hybrid) ───────────────────────────────

CANDIDATES = ["VC-M1", "VC-M3", "VC-M4"]


def test_review_sponsor_labels_from_index(built):
    labels = indexer.review_sponsor_labels()
    assert labels["rv-0017"]["label"] == "협찬"          # 무상 제공 문구
    assert labels["rv-0023"]["label"] == "실구매"        # "협찬 아닙니다"는 협찬 아님
    assert labels["rv-0007"]["label"] == "미정"          # 숨은 광고 → LLM 몫
    assert labels["rv-0017"]["model_id"] == indexer.catalog()["VC-004"]["model_id"]


def test_hybrid_excludes_sponsored_chunks(built):
    labels = indexer.review_sponsor_labels()
    r = retriever.search([QUERY], model_ids=CANDIDATES, mode="hybrid")
    assert r.chunks and all(labels[c["review_id"]]["label"] != "협찬" for c in r.chunks)
    assert r.excluded_sponsored == sum(1 for v in labels.values()
                                       if v["label"] == "협찬" and v["model_id"] in CANDIDATES)


def test_vector_baseline_does_not_exclude(built):
    labels = indexer.review_sponsor_labels()
    r = retriever.search([QUERY], model_ids=CANDIDATES, top_k=40, mode="vector")
    assert r.excluded_sponsored == 0
    assert any(labels[c["review_id"]]["label"] == "협찬" for c in r.chunks)


def test_hybrid_quota_prefers_organic(built):
    """VC-M3(슬림 S3)은 실구매 1·미정 1·협찬 1건 → 할당 2자리에 실구매 rv-0025가 들어가야 한다."""
    r = retriever.search([QUERY], model_ids=CANDIDATES, mode="hybrid")
    m3 = {c["review_id"] for c in r.chunks if c["model_id"] == "VC-M3"}
    assert "rv-0025" in m3 and "rv-0026" not in m3


def test_quota_select_priority():
    model = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B"}.get
    pri = {"a1": 1, "a2": 1, "a3": 0, "b1": 1, "b2": 0}.get
    ordered = ["a1", "a2", "a3", "b1", "b2"]
    # 할당은 우선순위(0 먼저), 같은 우선순위 안에서는 점수 순 / 나머지 자리는 점수 순
    assert retriever._quota_select(ordered, model, ["A", "B"], 1, 3, pri) == ["a3", "b2", "a1"]
