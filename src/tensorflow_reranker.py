"""Cross-encoder reranking without changing retrieved texts, metadata, or scores.

Example model: ``cross-encoder/ms-marco-MiniLM-L6-v2``, trained on query/passage
relevance with one output logit. Its PyTorch weights are converted to TensorFlow
on construction; all inference and ranking here use TensorFlow.
"""

from __future__ import annotations

import tensorflow as tf
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers import TFAutoModelForSequenceClassification

from src.schemas import RetrievedChunk


class TensorFlowReranker:
    """Score query/text pairs using a pretrained relevance cross-encoder.

    Supply a sequence-classification checkpoint trained for relevance, with a
    TensorFlow-supported architecture and convertible PyTorch weights. A single
    output is interpreted as a relevance logit (higher is better). For multiple
    outputs, config.id2label must explicitly identify exactly one 'relevant' or
    'relevance' class. Generic LABEL_0/LABEL_1 labels are deliberately rejected.
    Models are loaded once per instance, not on each rerank call.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        config = AutoConfig.from_pretrained(model_name)
        self._relevance_index = self._resolve_relevance_index(config)
        self._num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, from_pt=True
        )

    @staticmethod
    def _resolve_relevance_index(config: PretrainedConfig) -> int:
        """Resolve an explicit relevance output, rejecting ambiguous heads."""
        if config.num_labels == 1:
            return 0
        matches = [
            int(index)
            for index, label in (config.id2label or {}).items()
            if str(label).strip().casefold() in {"relevant", "relevance"}
        ]
        if len(matches) != 1 or not 0 <= matches[0] < config.num_labels:
            raise ValueError(
                "Cannot identify relevance class safely: multi-class models must "
                "have exactly one 'relevant' or 'relevance' entry in config.id2label."
            )
        return matches[0]

    def _relevance_scores(self, logits: tf.Tensor) -> tf.Tensor:
        """Use the configured relevance logit directly, not a guessed class."""
        logits = tf.cast(logits, tf.float32)
        tf.debugging.assert_rank(logits, 2)
        tf.debugging.assert_equal(tf.shape(logits)[1], self._num_labels)
        scores = logits[:, self._relevance_index]
        tf.debugging.assert_all_finite(scores, "Reranker produced non-finite scores.")
        return scores

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return top candidates by pair relevance, preserving original objects.

        Relevance logits are used only for ordering; RetrievedChunk.score remains
        the original retrieval score. The input list is not modified.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not chunks:
            return []

        score_batches: list[tf.Tensor] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            queries = [query] * len(batch)
            chunk_texts = [chunk.text for chunk in batch]
            inputs = self.tokenizer(
                queries,
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf",
            )
            outputs = self.model(**inputs, training=False)
            scores = self._relevance_scores(outputs.logits)
            tf.debugging.assert_equal(tf.shape(scores)[0], len(batch))
            score_batches.append(scores)

        scores = tf.concat(score_batches, axis=0)
        ranked = tf.math.top_k(scores, k=min(top_k, len(chunks)), sorted=True)
        return [chunks[int(index)] for index in ranked.indices]
