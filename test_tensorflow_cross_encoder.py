"""Run real retrieval and cross-encoder reranking without ingest or LLM calls."""

from unittest.mock import patch

import tensorflow as tf

from src.rag import retrieve
from src.store import close_client
from src.tensorflow_reranker import TensorFlowReranker


def main() -> None:
    query = "Fine-tuning mô hình ngôn ngữ lớn là gì?"
    print("Query:", query)
    candidates = retrieve(query, k=10)
    assert candidates, "No candidates found in the configured collection."
    original_scores = [chunk.score for chunk in candidates]
    original_indices = {id(chunk): index for index, chunk in enumerate(candidates)}

    print("\n=== BEFORE RERANKING: QDRANT TOP 10 ===")
    for rank, chunk in enumerate(candidates, start=1):
        assert chunk.text.strip()
        print(f"\nRank: {rank}")
        print("Qdrant score:", chunk.score)
        print("Page:", chunk.metadata.page)
        print("Chunk ID:", chunk.metadata.chunk_id)
        print("Content preview:", chunk.text[:300])

    reranker = TensorFlowReranker("cross-encoder/ms-marco-MiniLM-L6-v2")
    score_batches: list[tf.Tensor] = []
    score_function = reranker._relevance_scores

    def capture_scores(logits: tf.Tensor) -> tf.Tensor:
        # Observe real scores without replacing model inference or score logic.
        scores = score_function(logits)
        score_batches.append(scores)
        return scores

    # The production API returns original chunks, not relevance scores.
    # This test-only observer captures scores in candidate order during rerank.
    with patch.object(reranker, "_relevance_scores", side_effect=capture_scores):
        reranked = reranker.rerank(query, candidates, top_k=5)

    relevance_scores = tf.concat(score_batches, axis=0)
    assert int(tf.size(relevance_scores)) == len(candidates)
    assert bool(tf.reduce_all(tf.math.is_finite(relevance_scores)))
    assert len(reranked) == min(5, len(candidates))

    print("\n=== AFTER RERANKING: CROSS-ENCODER TOP 5 ===")
    for rank, chunk in enumerate(reranked, start=1):
        assert id(chunk) in original_indices, "Result is not an original candidate."
        assert chunk.text.strip()
        original_index = original_indices[id(chunk)]
        assert chunk.score == original_scores[original_index]
        print(f"\nNew rank: {rank}")
        print("Original Qdrant rank:", original_index + 1)
        print("Cross-encoder relevance score:", float(relevance_scores[original_index]))
        print("Original Qdrant score:", chunk.score)
        print("Page:", chunk.metadata.page)
        print("Chunk ID:", chunk.metadata.chunk_id)
        print("Content preview:", chunk.text[:300])

    print("\nTensorFlow cross-encoder test PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()
