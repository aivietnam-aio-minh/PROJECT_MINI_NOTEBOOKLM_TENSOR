"""Test existing Qdrant retrieval without ingesting data or calling an LLM."""

from src.rag import retrieve


def main() -> None:
    query = "Fine-tuning mô hình ngôn ngữ lớn là gì?"
    print("Query:", query)
    results = retrieve(query, k=5)
    print("Number of retrieved chunks:", len(results))

    assert len(results) > 0
    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", result.score)
        print("Page:", result.metadata.page)
        print("Filename:", result.metadata.filename)
        print("Chunk ID:", result.metadata.chunk_id)
        print("Content:")
        print(result.text[:500])
        assert result.text.strip()

    print("\nSemantic retrieval test PASSED")


if __name__ == "__main__":
    main()
