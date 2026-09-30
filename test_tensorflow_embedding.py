"""Inspect existing embeddings and compute cosine similarity with TensorFlow."""

import tensorflow as tf

from src.embeddings import get_embeddings


def main():
    sentences = [
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning uses neural networks.",
        "I like eating Vietnamese food.",
    ]

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(sentences)
    tensor = tf.convert_to_tensor(vectors, dtype=tf.float32)

    print("Tensor type:", type(tensor))
    print("Tensor dtype:", tensor.dtype)
    print("Embedding shape:", tensor.shape)
    print("First sentence, first 10 values:", tensor[0, :10])

    # The existing embeddings are L2-normalized, so cosine equals the dot product.
    similarity_1_2 = tf.reduce_sum(tensor[0] * tensor[1])
    similarity_1_3 = tf.reduce_sum(tensor[0] * tensor[2])

    print("Cosine similarity (sentence 1, sentence 2):", float(similarity_1_2))
    print("Cosine similarity (sentence 1, sentence 3):", float(similarity_1_3))


if __name__ == "__main__":
    main()
