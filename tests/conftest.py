"""테스트 공통 설정.

- 테스트 실행이 LangSmith trace를 남기지 않도록 끈다
  (src.tools가 .env를 override=False로 읽으므로 import 전에 여기서 설정한 값이 우선한다).
- 모든 테스트는 임시 Chroma 경로·임시 블로그 리뷰 파일·가짜 임베딩/리랭커를 쓴다.
  실제 data/chroma, data/reviews를 건드리지 않고, 1GB 모델을 내려받지 않는다.
"""

import hashlib
import math
import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest  # noqa: E402

from src import indexer, retriever  # noqa: E402


class FakeEmbedder:
    """글자 2-gram 해시 bag-of-words 임베딩 (결정적, 64차원, 정규화)."""

    name = "fake-embedder"
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        t = text.replace(" ", "")
        for i in range(len(t) - 1):
            h = int(hashlib.md5(t[i:i + 2].encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    """질의 토큰이 청크에 몇 개 들어 있는지로 채점. 호출 기록을 남긴다."""

    name = "fake-reranker"

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def score(self, query, texts):
        self.calls.append((query, len(texts)))
        q = set(indexer.tokenize(query))
        return [float(len(q & set(indexer.tokenize(t)))) for t in texts]


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(indexer, "BLOG_REVIEWS_PATH", tmp_path / "reviews" / "blog_reviews.json")
    monkeypatch.delenv("RETRIEVER_TOP_K", raising=False)
    indexer.set_embedder(FakeEmbedder())
    reranker = FakeReranker()
    retriever.set_reranker(reranker)
    yield reranker
    indexer.set_embedder(None)
    retriever.set_reranker(None)
