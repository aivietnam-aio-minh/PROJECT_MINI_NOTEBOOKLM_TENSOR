"""Local embeddings via Sentence-Transformers (no inference provider)."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


class LocalSentenceTransformerEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        vecs = _model().encode(texts, normalize_embeddings=True)
        tolist = getattr(vecs, "tolist", None)
        return tolist() if callable(tolist) else [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:  # type: ignore[override]
        vec = _model().encode([text], normalize_embeddings=True)[0]
        tolist = getattr(vec, "tolist", None)
        return tolist() if callable(tolist) else [float(x) for x in vec]


def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "tensorflow":
        from src.tensorflow_embeddings import TensorFlowEmbeddings

        return TensorFlowEmbeddings(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
        )
    return LocalSentenceTransformerEmbeddings()

