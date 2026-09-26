"""LLM invocation through Gemini API (google-genai)."""

from __future__ import annotations

from contextvars import ContextVar

from google import genai
from google.genai import types

from src.config import settings


_runtime_api_key: ContextVar[str | None] = ContextVar("runtime_gemini_api_key", default=None)

def set_runtime_gemini_api_key(api_key: str | None) -> None:
    _runtime_api_key.set(api_key.strip() if isinstance(api_key, str) else None)

def invoke_llm(prompt: str) -> str:
    """Invoke the configured chat model and return plain text.

    Args: prompt.
    Returns: assistant text content.
    Raises: RuntimeError if the API key is missing or response is empty.
    """
    api_key = _runtime_api_key.get() or settings.gemini_api_key
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Please provide it in the UI (or set GEMINI_API_KEY).")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_new_tokens,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini.")
    return text

