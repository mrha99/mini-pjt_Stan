"""BM25+벡터 앙상블, 리랭킹, 쿼리 확장 (패턴 3)

검색 모드는 설정값 RETRIEVER_MODE=vector|hybrid 로 고른다.

- vector (1차 기준선): 쿼리 확장·리랭킹 없이 원 질의(queries[0]) 1개로 Chroma 벡터 검색만 한다.
- hybrid (2차): 쿼리 확장(멀티 쿼리 3개) → 쿼리마다 BM25 + 벡터 검색 → RRF 가중 앙상블
                → 다국어 Cross-encoder 리랭킹 → 상위 k개.
                model_ids가 없으면 질의에 드러난 카테고리로 검색 범위를 좁힌다.
                후보 모델이 2개 이상이면 모델마다 최소 min(2, k ÷ 모델 수)개를 먼저 채운다
                (Top 3 카드마다 리뷰 근거 확보). vector 기준선은 점수 순 그대로.

두 모드 모두 model_ids(후보 모델)가 있으면 그 모델의 청크만 검색하고, 최종 반환 청크 수는
RETRIEVER_TOP_K(기본 8)로 같다 (1·2차 평가를 공정하게 비교). 블로그 청크(source=blog)는
답변 근거에서 빠지고 retrieve_blog_refs()로 "실제 블로그 참고 후기"만 따로 찾는다.

usage:
    python -m src.retriever "원룸 무선청소기 소음"                 # 현재 RETRIEVER_MODE로 검색
    python -m src.retriever --compare "원룸 무선청소기 소음"       # vector vs hybrid 비교
    python -m src.retriever --compare "질의" --model-ids VC-M1,VC-M2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Protocol

from dotenv import load_dotenv

from src import indexer

load_dotenv(indexer.ROOT / ".env", override=False)

DEFAULT_TOP_K = 8          # 최종 반환 청크 수 (vector·hybrid 공통)
CANDIDATE_K = 20           # hybrid: 쿼리·검색기마다 가져오는 후보 수
RERANK_POOL = 20           # hybrid: 앙상블 후 리랭커에 넘기는 후보 수 (CPU 리랭킹 약 3초)
RRF_K = 60                 # Reciprocal Rank Fusion 상수 (표준값)
VECTOR_WEIGHT = 0.6        # 앙상블 가중치: 의미 유사도(벡터)를 조금 더 신뢰
BM25_WEIGHT = 0.4          #               조건어 정확 일치(BM25)는 보조
MODEL_QUOTA_MAX = 2        # hybrid: 후보 모델이 2개 이상일 때 모델당 최소 할당 상한 (min(2, k ÷ 모델 수))

# 리랭커: BAAI/bge-reranker-v2-m3
# - bge-m3(XLM-RoBERTa 기반 다국어) 위에 학습한 Cross-encoder로 한국어 질의-문서 쌍을 직접 채점한다.
# - 질의와 청크를 함께 읽어 "소음이 작다" vs "소음이 크다" 같은 조건 방향성까지 구분한다
#   (임베딩 유사도만으로는 둘 다 '소음' 관련으로 가깝게 나옴).
# - 후보 20개 × 짧은 리뷰 청크라 CPU에서도 수 초 안에 끝난다.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 쿼리 확장용 조건어 사전: 사용자 조건어 → 리뷰에 실제로 쓰이는 표현
CONDITION_TERMS: dict[str, list[str]] = {
    "소음": ["조용", "시끄러", "저소음", "소리"],
    "원룸": ["자취", "좁은 집", "원룸 크기", "작은 방"],
    "흡입력": ["흡입", "빨아들", "먼지"],
    "무게": ["가벼", "무거", "손목"],
    "배터리": ["사용 시간", "충전", "배터리 시간"],
    "반려동물": ["털", "강아지", "고양이", "펫"],
    "아기": ["아이", "신생아", "육아", "아기방"],
    "냄새": ["악취", "냄새 차단", "초파리"],
    "세척": ["세척", "물때", "씻기", "관리"],
    "전기요금": ["전기세", "전력", "소비전력"],
    "용량": ["크기", "물통", "대용량"],
    "필터": ["필터 교체", "필터 가격", "소모품"],
    "가습량": ["가습", "습도", "건조"],
    "미세먼지": ["공기질", "미세먼지 수치", "정화"],
    "끓는 시간": ["빨리 끓", "끓는 속도", "보온"],
    "내구성": ["고장", "오래", "AS"],
    "가성비": ["가격 대비", "저렴", "만족"],
    "디자인": ["예쁘", "인테리어", "색상"],
}

_NOISE_RE = re.compile(
    r"\d[\d,.]*\s*(?:만\s*원|만원|원)\s*(?:이하|이상|미만|안쪽|대|정도|까지)?|"
    r"(?:추천해\s*줘|추천해\s*주세요|알려\s*줘|알려\s*주세요|해\s*줘|해\s*주세요|추천|골라\s*줘)|[?!.,]"
)


def _settings_mode() -> str:
    mode = os.getenv("RETRIEVER_MODE", "hybrid").strip().lower()
    return mode if mode in ("vector", "hybrid") else "hybrid"


def _settings_top_k() -> int:
    try:
        return max(1, int(os.getenv("RETRIEVER_TOP_K", DEFAULT_TOP_K)))
    except ValueError:
        return DEFAULT_TOP_K


# ─────────────────────────────── 쿼리 확장 ───────────────────────────────


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _NOISE_RE.sub(" ", text)).strip()


def extract_conditions(queries: list[str]) -> list[str]:
    """질의들에서 조건어를 뽑는다: 사전 키·동의어 일치 + 에이전트가 따로 넘긴 짧은 조건어.

    카테고리명은 먼저 지운다 ("무선청소기"의 '청소'가 조건어로 잡히지 않게).
    """
    text = " ".join(queries)
    for cat in _categories():
        text = text.replace(cat, " ")
    conds: list[str] = []
    for key, syns in CONDITION_TERMS.items():
        if key in text or any(s in text for s in syns):
            conds.append(key)
    for q in queries[1:]:
        q = _clean(q)
        if q and len(q) <= 15 and q not in conds and not any(q in c or c in q for c in conds):
            conds.append(q)
    return conds


def _categories() -> list[str]:
    return sorted({p["category"] for p in indexer.catalog().values()}, key=len, reverse=True)


def _category_of(text: str) -> str | None:
    for c in _categories():
        if c in text or c.replace("무선", "") in text:
            return c
    return None


def expand_queries(queries: list[str]) -> list[str]:
    """조건어를 붙인 멀티 쿼리 3개.

    1) 원 질의
    2) 핵심어 + 조건어 + "후기"            (BM25가 조건어 일치 리뷰를 잡도록)
    3) 카테고리 + 조건어의 리뷰 표현(동의어) (리뷰에 실제로 쓰는 말로 벡터 검색 보강)
    """
    base = queries[0].strip()
    core = _clean(base) or base
    conds = extract_conditions(queries)
    category = _category_of(" ".join(queries)) or core

    extra = [c for c in conds if c not in core]
    q2 = " ".join([core, *extra, "후기"]) if conds else f"{core} 실사용 후기"
    syns = [s for c in conds for s in CONDITION_TERMS.get(c, [c])[:2]]
    q3 = f"{category} {' '.join(syns)}".strip() if syns else f"{category} 장점 단점"

    out: list[str] = []
    for q in (base, q2, q3, f"{core} 장단점", f"{category} 실사용 만족"):
        if q not in out:
            out.append(q)
    return out[:3]


def rerank_query(queries: list[str]) -> str:
    """리랭커에 넣을 질의: 예산·요청 표현을 지운 원 질의 + (원 질의에 없는) 조건어."""
    core = _clean(queries[0]) or queries[0].strip()
    extra = [c for c in extract_conditions(queries) if c not in core]
    return f"{core} {' '.join(extra)}".strip()


# ─────────────────────────────── 리랭커 ───────────────────────────────


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=512, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts], batch_size=16,
                                                       show_progress_bar=False)]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def set_reranker(reranker: Reranker | None) -> None:
    global _reranker
    _reranker = reranker


# ─────────────────────────────── 검색 ───────────────────────────────


@dataclass
class RetrievalResult:
    mode: str
    chunks: list[dict]
    queries_used: list[str]
    rerank_query: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


def _where(model_ids: list[str] | None, source: str = "sample",
           categories: list[str] | None = None) -> dict:
    conds: list[dict] = [{"source": source}]
    if model_ids is not None:
        conds.append({"model_id": {"$in": list(model_ids)}})
    if categories:
        conds.append({"category": {"$in": list(categories)}})
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _chunk(doc_id: str, text: str, meta: dict, score: float, **extra) -> dict:
    return {"doc_id": doc_id, "text": text, "review_id": meta.get("review_id"),
            "model_id": meta.get("model_id"), "product_id": meta.get("product_id"),
            "mall": meta.get("mall"), "category": meta.get("category"),
            "source": meta.get("source"), "title": meta.get("title"), "link": meta.get("link"),
            "score": round(float(score), 4), **extra}


def _vector_search(col, query: str, k: int, where: dict) -> list[tuple[str, str, dict, float]]:
    n = min(k, col.count())
    if n == 0:
        return []
    res = col.query(query_embeddings=[indexer.get_embedder().embed_query(query)], n_results=n,
                    where=where, include=["documents", "metadatas", "distances"])
    return [(i, d, m, 1.0 - dist) for i, d, m, dist in
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def _allowed_positions(bm25: indexer.BM25Index, model_ids: list[str] | None,
                       category: str | None = None) -> set[int]:
    model_set = set(model_ids) if model_ids is not None else None
    return {i for i, m in enumerate(bm25.metadatas)
            if m.get("source") == "sample"
            and (model_set is None or m.get("model_id") in model_set)
            and (category is None or m.get("category") == category)}


def _quota_select(ordered: list[str], model_of, models: list[str], quota: int, limit: int) -> list[str]:
    """ordered(점수 내림차순)에서 모델마다 quota개를 먼저 고르고 나머지를 순서대로 채워 limit개."""
    if not quota:
        return ordered[:limit]
    picked: list[str] = []
    for m in models:
        picked += [c for c in ordered if model_of(c) == m][:quota]
    picked = picked[:limit]
    chosen = set(picked)
    picked += [c for c in ordered if c not in chosen][:limit - len(picked)]
    return picked


def search(queries: list[str], model_ids: list[str] | None = None, top_k: int | None = None,
           mode: str | None = None) -> RetrievalResult:
    """검색 본체. 반환 청크 dict: doc_id, text, review_id, model_id, product_id, mall,
    category, source, title, link, score (+hybrid: fusion_score)."""
    mode = mode or _settings_mode()
    k = top_k or _settings_top_k()
    queries = [q for q in (q.strip() for q in queries) if q]
    if not queries or (model_ids is not None and len(model_ids) == 0):
        return RetrievalResult(mode=mode, chunks=[], queries_used=queries)

    col = indexer.get_collection()
    where = _where(model_ids)
    t0 = time.perf_counter()

    if mode == "vector":
        hits = _vector_search(col, queries[0], k, where)
        chunks = [_chunk(i, d, m, s) for i, d, m, s in hits]
        return RetrievalResult(mode=mode, chunks=chunks, queries_used=[queries[0]],
                               timings={"vector": time.perf_counter() - t0})

    # hybrid
    expanded = expand_queries(queries)
    bm25 = indexer.get_bm25()
    # model_ids가 없고 질의에 카테고리가 드러나면 그 카테고리로 좁힌다 (다른 카테고리 리뷰 혼입 방지)
    category = _category_of(" ".join(queries)) if model_ids is None else None
    if category:
        where = _where(None, categories=[category])
    allowed = _allowed_positions(bm25, model_ids, category)
    fused: dict[str, float] = {}
    docs: dict[str, tuple[str, dict]] = {}
    for q in expanded:
        for rank, (cid, text, meta, _) in enumerate(_vector_search(col, q, CANDIDATE_K, where)):
            fused[cid] = fused.get(cid, 0.0) + VECTOR_WEIGHT / (RRF_K + rank + 1)
            docs[cid] = (text, meta)
        for rank, (pos, _) in enumerate(bm25.search(q, CANDIDATE_K, allowed)):
            cid = bm25.ids[pos]
            fused[cid] = fused.get(cid, 0.0) + BM25_WEIGHT / (RRF_K + rank + 1)
            docs[cid] = (bm25.documents[pos], bm25.metadatas[pos])
    # 후보 모델 2개 이상이면 모델별 최소 할당 (리뷰 많은 한 모델의 Top k 독차지 방지)
    models = list(dict.fromkeys(model_ids or []))
    quota = min(MODEL_QUOTA_MAX, k // len(models)) if len(models) > 1 else 0
    if quota:
        for m in models:  # 앙상블 후보에 없는 모델은 그 모델만 따로 벡터 검색해 후보를 보충
            have = sum(1 for c in fused if docs[c][1].get("model_id") == m)
            if have < quota:
                for rank, (cid, text, meta, _) in enumerate(
                        _vector_search(col, queries[0], quota, _where([m]))):
                    fused.setdefault(cid, VECTOR_WEIGHT / (RRF_K + CANDIDATE_K + rank + 1))
                    docs[cid] = (text, meta)
    t1 = time.perf_counter()

    by_fused = sorted(fused, key=lambda c: -fused[c])
    pool = _quota_select(by_fused, lambda c: docs[c][1].get("model_id"), models, quota, RERANK_POOL)
    rq = rerank_query(queries)
    scores = dict(zip(pool, get_reranker().score(rq, [docs[c][0] for c in pool]))) if pool else {}
    by_score = sorted(pool, key=lambda c: -scores[c])
    picked = _quota_select(by_score, lambda c: docs[c][1].get("model_id"), models, quota, k)
    ranked = sorted(((c, scores[c]) for c in picked), key=lambda x: -x[1])
    t2 = time.perf_counter()

    chunks = [_chunk(c, docs[c][0], docs[c][1], s, fusion_score=round(fused[c], 5))
              for c, s in ranked]
    return RetrievalResult(mode=mode, chunks=chunks, queries_used=expanded, rerank_query=rq,
                           timings={"ensemble": t1 - t0, "rerank": t2 - t1})


def retrieve(queries: list[str], model_ids: list[str] | None = None,
             top_k: int | None = None) -> list[dict]:
    """retrieve_reviews 도구가 호출하는 진입점 (현재 RETRIEVER_MODE, 근거 청크만)."""
    return search(queries, model_ids, top_k).chunks


def retrieve_blog_refs(queries: list[str], model_ids: list[str] | None = None,
                       k: int = 3) -> list[dict]:
    """'실제 블로그 참고 후기' 청크. 답변 근거(contexts)가 아니라 카드 아래 별도 표시용.

    블로그 청크는 상품과 연결되지 않으므로 model_ids가 있으면 그 모델들의 카테고리로만 좁힌다.
    """
    queries = [q for q in queries if q.strip()]
    if not queries:
        return []
    col = indexer.get_collection()
    if not col.get(where={"source": "blog"}, limit=1)["ids"]:
        return []
    cats = None
    if model_ids:
        cats = sorted({p["category"] for p in indexer.catalog().values() if p["model_id"] in model_ids})
    hits = _vector_search(col, rerank_query(queries), k, _where(None, source="blog", categories=cats))
    return [_chunk(i, d, m, s) for i, d, m, s in hits]


# ─────────────────────────────── CLI ───────────────────────────────


def _print_result(r: RetrievalResult) -> None:
    total = sum(r.timings.values())
    print(f"\n━━ {r.mode} · {len(r.chunks)}개 · {total:.2f}s "
          f"({', '.join(f'{k} {v:.2f}s' for k, v in r.timings.items())})")
    for i, q in enumerate(r.queries_used, 1):
        print(f"   쿼리{i}: {q}")
    if r.rerank_query:
        print(f"   리랭크 질의: {r.rerank_query}")
    for i, c in enumerate(r.chunks, 1):
        text = c["text"].replace("\n", " ")
        print(f"{i:>2}. {c['doc_id']:<11} {c['model_id'] or '-':<6} {c['score']:>7.3f}  "
              f"{text[:70]}{'…' if len(text) > 70 else ''}")


def compare(query: str, model_ids: list[str] | None = None, top_k: int | None = None) -> None:
    queries = [query]
    get_reranker()  # 모델 로딩 시간을 hybrid 검색 시간에서 뺀다
    indexer.get_embedder()
    results = {m: search(queries, model_ids, top_k, mode=m) for m in ("vector", "hybrid")}
    for r in results.values():
        _print_result(r)

    v_ids = [c["doc_id"] for c in results["vector"].chunks]
    h_ids = [c["doc_id"] for c in results["hybrid"].chunks]
    common = [d for d in h_ids if d in v_ids]
    print(f"\n━━ 비교: 공통 {len(common)}/{len(h_ids)}개")
    print(f"   hybrid에만: {', '.join(d for d in h_ids if d not in v_ids) or '-'}")
    print(f"   vector에만: {', '.join(d for d in v_ids if d not in h_ids) or '-'}")
    for m, r in results.items():
        dist: dict[str, int] = {}
        for c in r.chunks:
            dist[c["model_id"]] = dist.get(c["model_id"], 0) + 1
        print(f"   {m:<6} 모델 분포: " + ", ".join(f"{k} {v}" for k, v in dist.items()))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="리뷰 검색 (vector / hybrid)")
    parser.add_argument("query", nargs="?", help="검색 질의 (현재 RETRIEVER_MODE)")
    parser.add_argument("--compare", metavar="QUERY", help="같은 질의로 vector와 hybrid 결과 비교")
    parser.add_argument("--model-ids", help="후보 model_id 목록 (쉼표 구분)")
    parser.add_argument("--k", type=int, default=None, help="반환 청크 수 (기본 RETRIEVER_TOP_K=8)")
    args = parser.parse_args()
    model_ids = [m.strip() for m in args.model_ids.split(",")] if args.model_ids else None
    if args.compare:
        compare(args.compare, model_ids, args.k)
    elif args.query:
        _print_result(search([args.query], model_ids, args.k))
    else:
        parser.error("질의 또는 --compare QUERY가 필요합니다")


if __name__ == "__main__":
    main()
