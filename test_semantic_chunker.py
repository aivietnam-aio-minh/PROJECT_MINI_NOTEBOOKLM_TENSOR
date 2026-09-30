"""Integration test for semantic chunking with the project's real embeddings."""

from langchain_core.documents import Document

from src.embeddings import get_embeddings
from src.semantic_chunker import TensorFlowSemanticChunker


def main() -> None:
    document = Document(
        page_content=(
            "Machine learning is a field of artificial intelligence. "
            "Machine learning models can learn patterns from data. "
            "Deep learning uses neural networks with multiple layers. "
            "Vietnamese cuisine is known for its fresh ingredients. "
            "Pho is a popular traditional Vietnamese dish. "
            "Banh mi is another well-known Vietnamese food."
        ),
        metadata={
            "source": "test.pdf",
            "filename": "test.pdf",
            "page": 1,
        },
    )

    embeddings = get_embeddings()
    chunker = TensorFlowSemanticChunker(
        embeddings=embeddings,
        max_chunk_size=1000,
        threshold_k=1.0,
    )
    chunks = chunker.split_documents([document])

    print("Number of chunks:", len(chunks))
    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}")
        print("Content:", chunk.page_content)
        print("Metadata:", chunk.metadata)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.metadata["page"] == 1

    print("\nSemantic chunker test PASSED")


if __name__ == "__main__":
    main()
