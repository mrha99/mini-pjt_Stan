"""리뷰 수집·청크·Chroma 색인 (패턴 3)

- 리뷰를 청크로 나눠 Chroma(data/chroma/)에 색인한다. 청크 ID는 "{리뷰ID}#c{번호}".
- 청크 메타데이터: review_id, product_id, model_id, source(sample|blog), category (+ 표시용 필드).
  sample 리뷰 JSON에는 model_id가 없으므로 data/products.json에서 product_id로 찾아 넣는다.
  blog 리뷰는 특정 상품과 연결되지 않으므로 product_id·model_id 없이 source=blog, category만 넣는다
  (model_ids 필터 검색에서 자동으로 빠진다).
- BM25 인덱스는 Chroma의 청크 집합 그대로 메모리에 만들고, 색인이 추가되면 함께 다시 만든다.

usage:
    python -m src.indexer            # 초기 색인 (이미 색인된 ID는 건너뜀, run.sh에서 호출)
    python -m src.indexer --rebuild  # 컬렉션을 지우고 처음부터 색인
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.guardrails import mask_pii

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_PATH = ROOT / "data" / "products.json"
REVIEWS_DIR = ROOT / "data" / "reviews"
SAMPLE_REVIEWS_PATH = REVIEWS_DIR / "sample_reviews.json"
BLOG_REVIEWS_PATH = REVIEWS_DIR / "blog_reviews.json"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "reviews"

# 임베딩 모델: intfloat/multilingual-e5-base
# - 한국어를 포함한 100여 개 언어로 학습된 검색용 임베딩 (MIRACL·Mr.TyDi 한국어 검색 평가 포함)
# - 278M 파라미터·768차원으로 bge-m3(568M)보다 가벼워 CPU 컨테이너에서도 질의 임베딩이 빠르다.
#   (리랭커가 이미 bge-m3 계열 568M이라 임베딩까지 대형 모델을 쓰면 메모리·기동 시간이 부담)
# - "query: " / "passage: " 접두어 규약이 있어 질의와 리뷰를 비대칭으로 임베딩한다.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# 리뷰는 짧다(중간값 약 160자, 최대 약 270자). 200자 청크면 리뷰당 1~2개가 되어
# 한 청크가 한두 가지 관점(소음, 무게 등)만 담게 되고, 40자 겹침으로 문장 경계 손실을 줄인다.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "요. ", "다. ", "! ", "? ", " ", ""],
    keep_separator="end",
)


# ─────────────────────────────── 임베딩 ───────────────────────────────


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """sentence-transformers로 e5 모델을 불러 정규화된 임베딩을 만든다."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode([f"passage: {t}" for t in texts], batch_size=32,
                                  normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = E5Embedder()
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """테스트·실험용으로 임베딩 모델을 교체한다 (None이면 기본 모델로 복귀)."""
    global _embedder
    _embedder = embedder
    reset_cache()


# ─────────────────────────────── Chroma ───────────────────────────────

_clients: dict[str, object] = {}


def _client():
    import chromadb
    from chromadb.config import Settings

    key = str(CHROMA_DIR)
    if key not in _clients:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _clients[key] = chromadb.PersistentClient(
            path=key, settings=Settings(anonymized_telemetry=False))
    return _clients[key]


def get_collection():
    """리뷰 청크 컬렉션. 다른 임베딩 모델로 만든 색인이면 오류를 내 --rebuild를 안내한다."""
    client = _client()
    emb_name = _embedder.name if _embedder is not None else EMBEDDING_MODEL  # 모델 로딩 없이 이름만
    col = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine", "embedding_model": emb_name})
    built_with = (col.metadata or {}).get("embedding_model")
    if built_with and built_with != emb_name:
        raise RuntimeError(f"색인 임베딩 모델({built_with})과 현재 모델({emb_name})이 다릅니다. "
                           "python -m src.indexer --rebuild 로 다시 색인하세요.")
    return col


def reset_cache() -> None:
    """BM25·컬렉션 캐시를 비운다 (경로·모델 교체, 색인 추가 후)."""
    global _bm25, _labels_cache
    _bm25 = None
    _labels_cache = None


# ─────────────────────────────── 청크·메타데이터 ───────────────────────────────


@lru_cache(maxsize=4)
def _catalog(path: str) -> dict[str, dict]:
    products = json.loads(Path(path).read_text(encoding="utf-8"))["products"]
    return {p["product_id"]: p for p in products}


def catalog() -> dict[str, dict]:
    return _catalog(str(PRODUCTS_PATH))


def split_text(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    return [c.strip() for c in _splitter.split_text(text) if c.strip()] if text else []


def _clean_meta(meta: dict) -> dict:
    """Chroma 메타데이터는 None을 허용하지 않으므로 빈 값은 키째 뺀다."""
    return {k: v for k, v in meta.items() if v is not None and v != ""}


def review_to_chunks(review: dict) -> list[tuple[str, str, dict]]:
    """리뷰 1건 → [(chunk_id, text, metadata)]."""
    rid = review["review_id"]
    source = review.get("source") or ("blog" if rid.startswith("blog-") else "sample")
    if source == "blog":
        base = {"review_id": rid, "source": "blog", "category": review.get("category")}
    else:
        p = catalog().get(review.get("product_id", ""))
        if p is None:
            raise ValueError(f"{rid}: products.json에 없는 product_id {review.get('product_id')}")
        base = {"review_id": rid, "source": "sample", "product_id": p["product_id"],
                "model_id": p["model_id"], "category": p["category"], "mall": p["mall"]}
    base.update({"title": review.get("title"), "link": review.get("link"),
                 "date": review.get("date"), "rating": review.get("rating")})
    meta = _clean_meta(base)
    return [(f"{rid}#c{i}", text, {**meta, "chunk_index": i})
            for i, text in enumerate(split_text(review.get("body", "")))]


# ─────────────────────────────── 색인 ───────────────────────────────


def _existing_review_ids(col, review_ids: list[str]) -> set[str]:
    if not review_ids:
        return set()
    got = col.get(where={"review_id": {"$in": review_ids}}, include=["metadatas"])
    return {m["review_id"] for m in got["metadatas"]}


def add_reviews(reviews: list[dict], batch_size: int = 64) -> dict:
    """리뷰를 색인에 추가한다. 이미 색인된 review_id는 건너뛴다.

    returns {"added": 새로 색인한 리뷰 수, "skipped": 기존 리뷰 수, "chunks": 추가한 청크 수}
    """
    col = get_collection()
    unique = list({r["review_id"]: r for r in reviews}.values())
    existing = _existing_review_ids(col, [r["review_id"] for r in unique])
    new = [r for r in unique if r["review_id"] not in existing]

    chunks = [c for r in new for c in review_to_chunks(r)]
    for i in range(0, len(chunks), batch_size):
        part = chunks[i:i + batch_size]
        col.add(ids=[c[0] for c in part], documents=[c[1] for c in part],
                metadatas=[c[2] for c in part],
                embeddings=get_embedder().embed_documents([c[1] for c in part]))
    if chunks:
        reset_cache()  # BM25를 Chroma와 같은 청크 집합으로 다시 만들도록
    return {"added": len(new), "skipped": len(existing), "chunks": len(chunks)}


def _blog_id(link: str) -> str | None:
    m = re.search(r"(\d+)\D*$", link or "")
    return f"blog-{m.group(1)}" if m else None


def save_blog_reviews(reviews: list[dict]) -> int:
    """블로그 리뷰를 PII 마스킹 후 data/reviews/blog_reviews.json에 누적 저장 (SERVICE.md 정책 3).

    returns 새로 저장한 건수
    """
    stored: dict[str, dict] = {}
    if BLOG_REVIEWS_PATH.exists():
        doc = json.loads(BLOG_REVIEWS_PATH.read_text(encoding="utf-8"))
        stored = {r["review_id"]: r for r in doc.get("reviews", [])}
    added = 0
    for r in reviews:
        rid = r["review_id"]
        if rid in stored:
            continue
        stored[rid] = {
            "review_id": rid, "source": "blog", "category": r.get("category"),
            "title": mask_pii(r.get("title", "")), "body": mask_pii(r.get("body", "")),
            "author": r.get("author"), "date": r.get("date"), "link": r.get("link"),
        }
        added += 1
    if added:
        BLOG_REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "_meta": {
                "notice": "실제 블로그 참고 후기 (NAVER API HUB 블로그 검색 스니펫)",
                "description": "카테고리+조건어 검색 결과. 특정 상품의 근거가 아니며 전화번호·이메일·주소·계좌번호는 마스킹했다.",
                "is_fake": False,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "count": len(stored),
            },
            "reviews": list(stored.values()),
        }
        BLOG_REVIEWS_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def index_reviews(reviews: list[dict]) -> int:
    """fetch_reviews가 호출하는 진입점. 리뷰를 색인에 추가(기존 ID 건너뜀)하고,
    blog 리뷰는 마스킹 후 data/reviews/에 저장한다.

    returns 입력 리뷰 중 색인에 들어 있는 리뷰 수 (새로 추가 + 기존)
    """
    if not reviews:
        return 0
    blog, sample = [], []
    for r in reviews:
        rid = str(r.get("review_id", ""))
        if r.get("source") == "blog" or rid.startswith("blog-"):
            blog.append({**r, "source": "blog",
                         "review_id": rid if rid.startswith("blog-") else (_blog_id(r.get("link", "")) or rid),
                         "title": mask_pii(r.get("title", "")), "body": mask_pii(r.get("body", ""))})
        else:
            sample.append({**r, "source": "sample"})
    if blog:
        save_blog_reviews(blog)
    stats = add_reviews(sample + blog)
    return stats["added"] + stats["skipped"]


def load_review_files() -> list[dict]:
    """초기 색인 대상: 샘플 리뷰 전체 + (있으면) 저장된 블로그 리뷰."""
    reviews = [{**r, "source": "sample"} for r in
               json.loads(SAMPLE_REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]]
    if BLOG_REVIEWS_PATH.exists():
        reviews += json.loads(BLOG_REVIEWS_PATH.read_text(encoding="utf-8")).get("reviews", [])
    return reviews


def build_index(rebuild: bool = False) -> dict:
    if rebuild:
        client = _client()
        if COLLECTION_NAME in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION_NAME)
        reset_cache()
    stats = add_reviews(load_review_files())
    stats["total_chunks"] = get_collection().count()
    return stats


# ─────────────────────────────── BM25 ───────────────────────────────

_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """한국어 BM25 토큰화: 단어 + 한글 단어의 글자 2-gram.

    형태소 분석기 없이 조사 붙은 어절("소음이", "소음도")과 원형("소음")이 2-gram으로
    매칭되게 한다 (새 의존성 없이 Python 3.14에서 동작).
    """
    tokens: list[str] = []
    for w in _TOKEN_RE.findall((text or "").lower()):
        tokens.append(w)
        if len(w) > 2 and "가" <= w[0] <= "힣":
            tokens.extend(w[i:i + 2] for i in range(len(w) - 1))
    return tokens


class BM25Index:
    """Chroma 컬렉션의 전체 청크로 만든 BM25 인덱스."""

    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids, self.documents, self.metadatas = ids, documents, metadatas
        self._bm25 = BM25Okapi([tokenize(d) for d in documents]) if documents else None

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """[(청크 위치, 점수)] 상위 k. allowed가 있으면 그 위치만 대상."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        idx = range(len(scores)) if allowed is None else allowed
        ranked = sorted(((i, float(scores[i])) for i in idx), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > 0]


_bm25: BM25Index | None = None


def get_bm25() -> BM25Index:
    """Chroma와 같은 청크 집합의 BM25. 색인 추가나 다른 프로세스 변경(개수 불일치) 시 다시 만든다."""
    global _bm25
    col = get_collection()
    if _bm25 is None or len(_bm25) != col.count():
        got = col.get(include=["documents", "metadatas"])
        _bm25 = BM25Index(got["ids"], got["documents"], got["metadatas"])
    return _bm25


_labels_cache: tuple[int, dict[str, dict]] | None = None


def review_sponsor_labels() -> dict[str, dict]:
    """색인된 리뷰별 협찬 1차 규칙 라벨: review_id → {label: 협찬|실구매|미정, model_id, source}.

    BM25 인덱스(= Chroma 청크 집합)에서 리뷰 단위로 제목+청크를 이어 붙여 판정하므로
    다시 색인하지 않아도 되고, 색인이 바뀌면(BM25 재생성) 다시 계산한다.
    hybrid 검색의 협찬 청크 제외와 추천 카드의 리뷰 구분 집계에 쓴다.
    """
    global _labels_cache
    from src.chains import classify_rule

    bm25 = get_bm25()
    if _labels_cache is not None and _labels_cache[0] == id(bm25):
        return _labels_cache[1]
    parts: dict[str, list[tuple[int, str]]] = {}
    meta_of: dict[str, dict] = {}
    for text, meta in zip(bm25.documents, bm25.metadatas):
        rid = meta.get("review_id")
        if not rid:
            continue
        parts.setdefault(rid, []).append((int(meta.get("chunk_index", 0)), text))
        meta_of.setdefault(rid, meta)
    labels = {}
    for rid, chunks in parts.items():
        meta = meta_of[rid]
        body = " ".join(t for _, t in sorted(chunks))
        rule = classify_rule(f"{meta.get('title', '')}\n{body}")
        labels[rid] = {"label": rule.label or "미정", "model_id": meta.get("model_id"),
                       "source": meta.get("source")}
    _labels_cache = (id(bm25), labels)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="리뷰 청크 Chroma 색인")
    parser.add_argument("--rebuild", action="store_true", help="컬렉션을 지우고 처음부터 색인")
    args = parser.parse_args()
    stats = build_index(rebuild=args.rebuild)
    print(f"새로 색인 {stats['added']}건(청크 {stats['chunks']}개), 기존 {stats['skipped']}건 건너뜀, "
          f"전체 청크 {stats['total_chunks']}개 → {CHROMA_DIR}")


if __name__ == "__main__":
    main()
