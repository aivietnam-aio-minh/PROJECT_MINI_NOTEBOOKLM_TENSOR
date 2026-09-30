"""Sentence-preserving semantic chunking with TensorFlow similarities."""

from __future__ import annotations

import re

import tensorflow as tf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TensorFlowSemanticChunker:
    """Reuse an injected, L2-normalized embedder to split documents."""

    def __init__(
        self,
        embeddings: Embeddings,
        max_chunk_size: int = 1500,
        threshold_k: float = 1.0,
        min_chunk_size: int = 100,
    ) -> None:
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive.")
        if not 0 <= min_chunk_size <= max_chunk_size:
            raise ValueError("min_chunk_size must be between 0 and max_chunk_size.")
        self.embeddings = embeddings
        self.max_chunk_size = max_chunk_size
        self.threshold_k = threshold_k
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on whitespace after . ! ?, preserving punctuation."""
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]

    def _semantic_boundaries(self, sentences: list[str]) -> tf.Tensor:
        embeddings = self.embeddings.embed_documents(sentences)
        tensor = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        left_embeddings = tensor[:-1]
        right_embeddings = tensor[1:]
        similarities = tf.reduce_sum(left_embeddings * right_embeddings, axis=1)

        mean_similarity = tf.reduce_mean(similarities)
        std_similarity = tf.math.reduce_std(similarities)
        k = tf.cast(self.threshold_k, tf.float32)
        threshold = mean_similarity - k * std_similarity
        boundary_mask = similarities < threshold
        return boundary_mask

    def _join_chunks(self, sentences: list[str], boundary_mask: tf.Tensor) -> list[str]:
        chunks: list[str] = []
        current_chunk = ""
        for index, sentence in enumerate(sentences):
            candidate = f"{current_chunk} {sentence}" if current_chunk else sentence
            semantic_boundary = index > 0 and bool(boundary_mask[index - 1])
            size_exceeded = len(candidate) > self.max_chunk_size
            should_cut = semantic_boundary or size_exceeded

            # Keep an oversized sentence intact; never emit an empty chunk.
            if current_chunk and should_cut:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = candidate

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _merge_tiny_chunks(self, chunks: list[str]) -> list[str]:
        """Merge small neighbors without exceeding the limit or reordering text."""
        merged = list(chunks)
        index = 0
        while index < len(merged):
            if len(merged[index].strip()) >= self.min_chunk_size:
                index += 1
                continue

            if index > 0:
                candidate = f"{merged[index - 1]} {merged[index]}"
                if len(candidate) <= self.max_chunk_size:
                    merged[index - 1] = candidate
                    del merged[index]
                    index -= 1
                    continue

            if index + 1 < len(merged):
                candidate = f"{merged[index]} {merged[index + 1]}"
                if len(candidate) <= self.max_chunk_size:
                    merged[index] = candidate
                    del merged[index + 1]
                    continue

            index += 1
        return merged

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            sentences = self._split_sentences(document.page_content)
            if not sentences:
                continue
            if len(sentences) == 1:
                texts = sentences
            else:
                boundary_mask = self._semantic_boundaries(sentences)
                initial_chunks = self._join_chunks(sentences, boundary_mask)
                texts = self._merge_tiny_chunks(initial_chunks)

            chunks.extend(
                Document(page_content=text, metadata=dict(document.metadata))
                for text in texts
            )
        return chunks
