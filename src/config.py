"""Application configuration for Hugging Face Spaces.

Defaults live in code; a small set of values can be overridden via env vars.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant_tensorflow")
    qdrant_collection: str = "rag_chunks_tensorflow"

    chunk_size: int = Field(default=1500, ge=500)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)

    llm_provider: Literal["gemini"] = "gemini"
    llm_model: str = "gemini-flash-lite-latest"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_new_tokens: int = Field(default=10000, ge=1, le=20000)

    embedding_provider: Literal["local", "tensorflow"] = "tensorflow"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    embedding_max_length: int = Field(default=128, ge=1, le=512)

    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    quiz_default_count: int = Field(default=8, ge=1, le=50)
    flashcards_default_count: int = Field(default=15, ge=1, le=100)

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

