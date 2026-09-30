"""Test semantic build_chunks on an existing PDF without indexing it."""

from pathlib import Path

from src.config import settings
from src.indexing import build_chunks


def main() -> None:
    pdfs = sorted(Path("data").rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF found in data/. Run from the project root with a PDF in data/.")
    if settings.chunking_strategy != "semantic" or settings.embedding_provider != "tensorflow":
        raise ValueError(
            "Set RAG_CHUNKING_STRATEGY=semantic and RAG_EMBEDDING_PROVIDER=tensorflow "
            "to run this integration test."
        )

    print("PDF:", pdfs[0])
    chunks = build_chunks([pdfs[0]])
    lengths = [len(chunk.page_content.strip()) for chunk in chunks]
    print("Number of chunks:", len(chunks))
    assert len(chunks) > 0
    print("Minimum chunk length:", min(lengths))
    print("Maximum chunk length:", max(lengths))
    print("Average chunk length:", sum(lengths) / len(lengths))

    tiny_chunks = [
        (index, chunk, length)
        for index, (chunk, length) in enumerate(zip(chunks, lengths), start=1)
        if length < 100
    ]
    print("Tiny chunks (<100 chars):", len(tiny_chunks))
    for index, chunk, length in tiny_chunks:
        print(f"\nTiny chunk {index}")
        print("Length:", length)
        print("Content:", chunk.page_content)

    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\nChunk {index}")
        print("Length:", len(chunk.page_content))
        print("Content preview:", chunk.page_content[:300])
        print("Metadata:", chunk.metadata)

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "document_id" in chunk.metadata
        assert "filename" in chunk.metadata
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert "chunk_id" in chunk.metadata

    print("\nbuild_chunks semantic integration PASSED")


if __name__ == "__main__":
    main()
