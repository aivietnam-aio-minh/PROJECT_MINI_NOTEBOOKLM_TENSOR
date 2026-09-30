"""TensorFlow inference for the project's multilingual MiniLM sentence encoder.

The source checkpoint is PyTorch; Transformers converts its weights on load.
Mean pooling with an attention mask reproduces the model's sentence pooling.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
import tensorflow as tf


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    try:
        from transformers import AutoTokenizer, TFAutoModel
        
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow backend requires requirements-tensorflow.txt. "
            "Use its pinned Transformers 4.x environment."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Transformers detects the PyTorch format in safetensors metadata and
    # converts it to TF. Explicit from_pt=True would look for a .bin file.
    model = TFAutoModel.from_pretrained(model_name, use_safetensors=True)
    return tf, tokenizer, model


@tf.function(reduce_retracing=True)
def masked_mean_pool(token_embeddings, attention_mask):
    """Ignore padding and return unit-length sentence vectors."""

    tokens = tf.cast(token_embeddings, tf.float32)
    mask = tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=-1)
    summed = tf.reduce_sum(tokens * mask, axis=1)
    counts = tf.reduce_sum(mask, axis=1)
    mean = tf.math.divide_no_nan(summed, counts)
    return tf.linalg.l2_normalize(mean, axis=1)


class TensorFlowEmbeddings(Embeddings):
    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512):
        if batch_size < 1 or not 1 <= max_length <= 512:
            raise ValueError("Invalid embedding batch size or token limit.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        _, tokenizer, model = _load_model(self.model_name)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            inputs = tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf",
            )
            outputs = model(**inputs, training=False)
            pooled = masked_mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            vectors.extend(pooled.numpy().tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
