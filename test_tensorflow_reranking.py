"""Experiment with TensorFlow cosine reranking of existing retrieval results."""

import tensorflow as tf

from src.rag import retrieve
from src.embeddings import get_embeddings
from src.store import close_client


def main() -> None:
    query = "Fine-tuning mô hình ngôn ngữ lớn là gì?"
    print("Query:", query)
    candidates = retrieve(query, k=10)
    assert len(candidates) > 0, "Expected retrieved candidates."

    embeddings = get_embeddings()
    query_tensor = tf.convert_to_tensor(
        embeddings.embed_query(query), dtype=tf.float32
    )
    candidate_tensor = tf.convert_to_tensor(
        embeddings.embed_documents([candidate.text for candidate in candidates]),
        dtype=tf.float32,
    )

    # Existing embeddings are L2-normalized, so dot products are cosine scores.
    scores = tf.linalg.matvec(candidate_tensor, query_tensor)
    assert bool(tf.reduce_all(tf.math.is_finite(scores))), "Non-finite TensorFlow scores."
    ranked = tf.math.top_k(scores, k=len(candidates), sorted=True)
    reranked = [int(index) for index in ranked.indices]
    assert len(reranked) == len(candidates)

    def print_result(rank: int, original_index: int) -> None:
        candidate = candidates[original_index]
        print(f"\nRank {rank}")
        print("Original rank:", original_index + 1)
        print("Qdrant score:", candidate.score)
        print("TensorFlow score:", float(scores[original_index]))
        print("Page:", candidate.metadata.page)
        print("Chunk ID:", candidate.metadata.chunk_id)
        print("Content preview:", candidate.text[:300])

    print("\n=== Qdrant ranking ===")
    for rank, original_index in enumerate(range(len(candidates)), start=1):
        print_result(rank, original_index)

    print("\n=== TensorFlow reranking ===")
    for rank, original_index in enumerate(reranked, start=1):
        print_result(rank, original_index)

    print("\nTensorFlow reranking experiment PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()
