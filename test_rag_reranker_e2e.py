"""Validate the real public RAG flow, including the configured Gemini API."""

from unittest.mock import patch

from src import rag
from src.config import settings
from src.schemas import RetrievedChunk
from src.store import close_client
from src.tensorflow_reranker import TensorFlowReranker


def main() -> None:
    query = "Fine-tuning mô hình ngôn ngữ lớn là gì?"
    candidates: list[RetrievedChunk] = []
    final_chunks: list[RetrievedChunk] = []
    snapshots: dict[int, dict] = {}
    real_retrieve = rag.retrieve
    real_rerank = TensorFlowReranker.rerank

    def observe_retrieve(*args, **kwargs):
        assert kwargs["k"] == 10
        found = real_retrieve(*args, **kwargs)
        candidates.extend(found)
        snapshots.update({id(chunk): chunk.model_dump() for chunk in found})
        return found

    def observe_rerank(self, query, chunks, top_k=5):
        assert [id(chunk) for chunk in chunks] == [id(chunk) for chunk in candidates]
        ranked = real_rerank(self, query, chunks, top_k=top_k)
        final_chunks.extend(ranked)
        return ranked

    print("Query:", query, sep="\n")
    # Observers call the real implementations; no model or LLM is stubbed.
    # Only this test process enables reranking; .env is left unchanged.
    with patch.multiple(
        settings, reranker_enabled=True, reranker_candidate_k=10,
        reranker_top_k=5, embedding_provider="tensorflow",
    ), patch.object(rag, "retrieve", side_effect=observe_retrieve), patch.object(
        TensorFlowReranker, "rerank", new=observe_rerank
    ), patch.object(rag, "prepare_context", wraps=rag.prepare_context) as context, patch.object(
        rag, "render_prompt", wraps=rag.render_prompt
    ) as prompt, patch.object(rag, "invoke_llm", wraps=rag.invoke_llm) as llm:
        result = rag.answer(query)
        context.assert_called_once()
        prompt.assert_called_once()
        llm.assert_called_once()
        assert result.answer.strip()
        assert result.chunks and len(result.chunks) <= settings.reranker_top_k
        assert len(result.chunks) == min(5, len(candidates))
        assert [id(c) for c in result.chunks] == [id(c) for c in final_chunks]
        assert [id(c) for c in prompt.call_args.kwargs["chunks"]] == [id(c) for c in final_chunks]
        assert result.citations

    print("\nCandidates retrieved:", len(candidates))
    print("Final chunks:", len(result.chunks))
    print("\nAnswer:", result.answer, sep="\n")
    print("\nFinal chunks sent to Gemini:")
    for rank, chunk in enumerate(result.chunks, start=1):
        assert chunk.text.strip()
        assert id(chunk) in snapshots
        assert chunk.model_dump() == snapshots[id(chunk)]
        print(f"\nFinal rank: {rank}")
        print("Original Qdrant rank:", next(i for i, c in enumerate(candidates, 1) if c is chunk))
        print("Original Qdrant score:", chunk.score)
        print("Page:", chunk.metadata.page)
        print("Chunk ID:", chunk.metadata.chunk_id)
        print("Content preview:", chunk.text[:300])

    print("\nCitations:")
    for citation in result.citations:
        assert 1 <= citation.source_index <= len(final_chunks)
        chunk = final_chunks[citation.source_index - 1]
        assert citation.source_marker == f"S{citation.source_index}"
        assert citation.chunk_id == chunk.metadata.chunk_id
        assert citation.filename == chunk.metadata.filename
        assert citation.page == chunk.metadata.page
        assert citation.source_text == chunk.text.strip()
        print(f"{citation.source_marker} | {citation.filename} | page={citation.page} | chunk_id={citation.chunk_id}")

    print("\nTensorFlow reranked RAG end-to-end PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()
