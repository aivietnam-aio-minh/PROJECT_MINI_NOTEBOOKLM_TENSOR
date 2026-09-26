"""Run real encoder inference without calling Gemini: python check_tensorflow.py."""

import numpy as np
import tensorflow as tf

from src.embeddings import get_embeddings
from src.tensorflow_embeddings import TensorFlowEmbeddings, masked_mean_pool


def main():
    # Padding must not affect the sentence representation.
    pooled = masked_mean_pool(
        tf.constant([[[3., 4.], [999., 999.]]]), tf.constant([[1, 0]])
    ).numpy()
    np.testing.assert_allclose(pooled, [[0.6, 0.8]], atol=1e-6)
    embedder = get_embeddings()
    assert isinstance(embedder, TensorFlowEmbeddings), "Select the tensorflow backend."
    assert embedder.embed_documents([]) == []
    texts = ["Học máy giúp máy tính học từ dữ liệu.", "Hôm nay trời mưa."]
    vectors = np.asarray(embedder.embed_documents(texts))
    assert vectors.shape == (2, 384), vectors.shape
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    # Batching/padding must not change query vs document embeddings.
    query = np.asarray(embedder.embed_query(texts[0]))
    np.testing.assert_allclose(query, vectors[0], atol=1e-5)
    assert int(np.argmax(vectors @ query)) == 0
    print("PASS: TensorFlow inference, 384 dimensions, normalization, padding, retrieval.")


if __name__ == "__main__":
    main()
