"""Test the existing RAG flow against ingested chunks using configured Gemini credentials."""

from src.rag import answer


def main() -> None:
    query = "Fine-tuning mô hình ngôn ngữ lớn là gì?"
    print("Query:")
    print(query)

    result = answer(query, k=5)

    print("\nAnswer:")
    print(result.answer)
    print("\nCitations:")
    for citation in result.citations:
        print(citation.model_dump())

    assert result.answer.strip(), "Answer must not be empty."
    assert result.citations, "Expected citations from retrieved sources."
    assert result.chunks, "Expected retrieved context chunks."
    for citation in result.citations:
        assert citation.filename.strip(), "Citation must identify a source file."
        assert citation.source_text and citation.source_text.strip(), "Citation must contain source text."
    for chunk in result.chunks:
        assert chunk.text.strip(), "Retrieved context must not be empty."

    print("\nSemantic RAG end-to-end test PASSED")


if __name__ == "__main__":
    main()
