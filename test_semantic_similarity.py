"""Compute adjacent sentence similarities using vectorized TensorFlow operations."""

import tensorflow as tf

from src.embeddings import get_embeddings


def main():
    sentences = [
        "Machine learning is a field of artificial intelligence.",
        "Machine learning models can learn patterns from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Vietnamese cuisine is known for its fresh ingredients.",
        "Pho is a popular traditional Vietnamese dish.",
        "Banh mi is another well-known Vietnamese food.",
    ]

    embeddings = get_embeddings().embed_documents(sentences)
    tensor = tf.convert_to_tensor(embeddings, dtype=tf.float32)

    left_embeddings = tensor[:-1]
    right_embeddings = tensor[1:]
    # L2-normalized vectors: compute all five cosine similarities at once.
    similarities = tf.reduce_sum(left_embeddings * right_embeddings, axis=1)

    mean_similarity = tf.reduce_mean(similarities)
    std_similarity = tf.math.reduce_std(similarities)
    k = tf.constant(1.0, dtype=tf.float32)
    threshold = mean_similarity - k * std_similarity
    boundary_mask = similarities < threshold
    boundary_indices = tf.reshape(tf.where(boundary_mask), [-1])

    print("Embeddings shape:", tensor.shape)
    print("Left embeddings shape:", left_embeddings.shape)
    print("Right embeddings shape:", right_embeddings.shape)
    print("Similarities shape:", similarities.shape)

    # This loop only displays the results; similarity computation is vectorized.
    for index, similarity in enumerate(similarities):
        print(f"\nSentence {index + 1}: {sentences[index]}")
        print(f"Sentence {index + 2}: {sentences[index + 1]}")
        print(f"Cosine similarity: {float(similarity):.6f}")

    print("\nMean similarity:", float(mean_similarity))
    print("Standard deviation:", float(std_similarity))
    print("Adaptive threshold:", float(threshold))
    print("Boundary mask:", boundary_mask)
    print("Boundary indices:", boundary_indices)

    max_chunk_size = 180
    chunks = []
    current_chunk = ""
    for index, sentence in enumerate(sentences):
        candidate = f"{current_chunk} {sentence}" if current_chunk else sentence
        semantic_boundary = index > 0 and bool(boundary_mask[index - 1])
        size_exceeded = len(candidate) > max_chunk_size
        should_cut = semantic_boundary or size_exceeded

        # Keep sentences intact and never emit an empty chunk.
        if current_chunk and should_cut:
            chunks.append(current_chunk)
            if semantic_boundary and size_exceeded:
                reason = "semantic + max_size"
            elif semantic_boundary:
                reason = "semantic"
            else:
                reason = "max_size"
            print(f"\nCUT before sentence {index + 1}")
            print(f"Reason: {reason}")
            current_chunk = sentence
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    print("\nSemantic chunks:")
    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index} ({len(chunk)} chars):")
        print(chunk)


if __name__ == "__main__":
    main()
