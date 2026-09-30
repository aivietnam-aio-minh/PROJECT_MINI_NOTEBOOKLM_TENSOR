"""Run real encoder inference without calling Gemini: python check_tensorflow.py."""

import numpy as np
import tensorflow as tf
from unittest.mock import patch

from src.embeddings import LocalSentenceTransformerEmbeddings, get_embeddings
from src.tensorflow_embeddings import TensorFlowEmbeddings, masked_mean_pool


def main():
    # Padding must not affect the sentence representation.
    pooled = masked_mean_pool(
        tf.constant([[[3., 4.], [999., 999.]]]), tf.constant([[1, 0]])
    ).numpy()
    np.testing.assert_allclose(pooled, [[0.6, 0.8]], atol=1e-6)
    empty_pool = masked_mean_pool(
        tf.ones((2, 3, 2)), tf.zeros((2, 3), dtype=tf.int32)
    ).numpy()
    np.testing.assert_array_equal(empty_pool, np.zeros((2, 2)))
    with patch("src.embeddings._model") as local_model:
        assert LocalSentenceTransformerEmbeddings().embed_documents([]) == []
        local_model.assert_not_called()
    with patch("src.tensorflow_embeddings._load_model") as tf_model:
        assert TensorFlowEmbeddings("unused").embed_documents([]) == []
        tf_model.assert_not_called()
    embedder = get_embeddings()
    assert get_embeddings() is embedder
    assert isinstance(embedder, TensorFlowEmbeddings), "Select the tensorflow backend."
    assert embedder.embed_documents([]) == []
    texts = ["Học máy giúp máy tính học từ dữ liệu.", "Hôm nay trời mưa."]
    result = embedder.embed_documents(texts)
    assert isinstance(result, list)
    assert all(isinstance(row, list) for row in result)
    assert all(isinstance(value, float) for row in result for value in row)
    vectors = np.asarray(result)
    assert vectors.shape == (2, 384), vectors.shape
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    # Batching/padding must not change query vs document embeddings.
    query = np.asarray(embedder.embed_query(texts[0]))
    np.testing.assert_allclose(query, vectors[0], atol=1e-5)
    assert int(np.argmax(vectors @ query)) == 0
    # Different content after token 128 must still contribute to the vector.
    if embedder.max_length >= 384:
        prefix = "hello " * 140
        long_vectors = np.asarray(embedder.embed_documents([
            prefix + "machine learning " * 40,
            prefix + "rainy weather " * 40,
        ]))
        assert not np.allclose(long_vectors[0], long_vectors[1], atol=1e-5)
    print("PASS: TensorFlow inference, 384 dimensions, normalization, padding, retrieval.")


if __name__ == "__main__":
    main()
