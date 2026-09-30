"""Exercise the production context path with real Qdrant and no Gemini calls."""

from unittest.mock import patch

from src import rag
from src.config import Settings, settings
from src.schemas import RetrievedChunk
from src.store import close_client


def check_edge_cases() -> None:
    """Check disabled/empty paths, k forwarding, and invalid configuration."""
    try:
        Settings(_env_file=None, reranker_candidate_k=2, reranker_top_k=5)
    except ValueError as exc:
        assert "reranker_candidate_k must be >= reranker_top_k" in str(exc)
    else:
        raise AssertionError("Invalid reranker config accepted.")

    filters = {"filename": "test.pdf"}
    with patch.object(settings, "reranker_enabled", False), patch.object(
        rag, "retrieve", return_value=[]
    ) as retrieve_mock, patch.object(rag, "_get_reranker") as loader:
        assert rag.prepare_context("query", k=3, filters=filters, collection_name="test") == []
        retrieve_mock.assert_called_once_with("query", k=3, filters=filters, collection_name="test")
        loader.assert_not_called()

    with patch.object(settings, "reranker_enabled", True), patch.object(
        rag, "retrieve", return_value=[]
    ) as retrieve_mock, patch.object(rag, "_get_reranker") as loader:
        assert rag.prepare_context("query", k=12) == []
        assert retrieve_mock.call_args.kwargs["k"] >= 12
        loader.assert_not_called()
        with patch.object(rag, "invoke_llm", side_effect=AssertionError("LLM must not run")):
            result = rag.answer("query")
            assert result.chunks == [] and result.citations == []


def print_chunks(title: str, chunks: list[RetrievedChunk]) -> None:
    print(f"\n{title}")
    for rank, chunk in enumerate(chunks, start=1):
        print(f"\nRank: {rank}")
        print("Original Qdrant score:", chunk.score)
        print("Page:", chunk.metadata.page)
        print("Chunk ID:", chunk.metadata.chunk_id)
        print("Content preview:", chunk.text[:300])


def main() -> None:
    check_edge_cases()
    query = "Fine-tuning mô hình ngôn ngữ lớn là gì?"
    candidates: list[RetrievedChunk] = []
    snapshots: dict[int, dict] = {}
    real_retrieve = rag.retrieve

    def capture_candidates(*args: object, **kwargs: object) -> list[RetrievedChunk]:
        retrieved = real_retrieve(*args, **kwargs)
        candidates.extend(retrieved)
        snapshots.update({id(chunk): chunk.model_dump() for chunk in retrieved})
        return retrieved

    # Override only this process's settings, leaving .env and production defaults intact.
    with patch.multiple(
        settings,
        reranker_enabled=True,
        reranker_candidate_k=10,
        reranker_top_k=5,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        reranker_batch_size=8,
    ), patch.object(rag, "invoke_llm", side_effect=AssertionError("LLM must not run")):
        with patch.object(rag, "retrieve", side_effect=capture_candidates):
            final_chunks = rag.prepare_context(query)

        print_chunks("=== QDRANT CANDIDATES ===", candidates)
        print_chunks("=== FINAL CONTEXT AFTER RERANKING ===", final_chunks)
        assert final_chunks
        assert len(final_chunks) == min(settings.reranker_top_k, len(candidates))
        for chunk in final_chunks:
            assert id(chunk) in snapshots
            assert chunk.text.strip()
            assert chunk.metadata.chunk_id and chunk.metadata.filename
            assert chunk.model_dump() == snapshots[id(chunk)]

        # Confirm reuse of the loaded model and behavior with fewer candidates.
        cached = rag._get_reranker()
        assert rag._get_reranker() is cached
        assert len(cached.rerank(query, candidates[:2], top_k=5)) == min(2, len(candidates))

        # Validate answer's context/citations wiring with a stub, never Gemini.
        with patch.object(rag, "prepare_context", return_value=final_chunks) as context, patch.object(
            rag, "invoke_llm", return_value="Stub answer for wiring validation."
        ), patch.object(rag, "render_prompt", return_value="stub prompt") as prompt:
            result = rag.answer(query, k=3, filters={"page": 1}, collection_name="test")
            context.assert_called_once_with(query, k=3, filters={"page": 1}, collection_name="test")
            assert prompt.call_args.kwargs["chunks"] is final_chunks
            assert [c.chunk_id for c in result.citations] == [c.metadata.chunk_id for c in final_chunks]
            assert [c.source_marker for c in result.citations] == [f"S{i}" for i in range(1, len(final_chunks) + 1)]

    print("\nRAG reranker integration PASSED (no Gemini calls)")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()
