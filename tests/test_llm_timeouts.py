import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.llm_service import LLMService


def test_llm_service_reads_configured_timeouts(monkeypatch):
    """LLMService must read LLM_TIMEOUT_SECONDS and LLM_STREAM_FIRST_TOKEN_TIMEOUT from environment."""
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_STREAM_FIRST_TOKEN_TIMEOUT", "15")

    service = LLMService(token="hf_mock_token")
    assert service.timeout_seconds == 45.0
    assert service.stream_first_token_timeout == 15.0


def test_llm_service_timeout_triggers_exhaustion_and_groq_fallback():
    """Hugging Face timeout exceptions must trigger Groq fallback."""
    service = LLMService(token="hf_mock_token")
    assert not service.hf_exhausted

    # Simulate timeout error from HF client
    timeout_err = TimeoutError("Request timed out after 60.0 seconds")
    is_exhausted = service._is_hf_exhausted(timeout_err)

    assert is_exhausted is True
    assert service.hf_exhausted is True


@pytest.mark.asyncio
async def test_streaming_first_token_timeout_triggers_groq_fallback():
    """When HF stream hangs on first token, it times out and falls back to Groq stream."""
    service = LLMService(token="hf_mock_token")
    service.stream_first_token_timeout = 0.05  # 50ms for fast test

    # Mock HF client stream that hangs indefinitely
    async def hanging_stream():
        await asyncio.sleep(2.0)
        yield MagicMock()

    service.client = MagicMock()
    service.client.chat_completion = AsyncMock(return_value=hanging_stream())

    # Mock Groq client fallback that responds immediately
    mock_choice = MagicMock()
    mock_choice.delta.content = "Answer from Groq fallback"
    mock_groq_chunk = MagicMock()
    mock_groq_chunk.choices = [mock_choice]

    async def fast_groq_stream():
        yield mock_groq_chunk

    service.groq_client = MagicMock()
    service.groq_client.chat.completions.create = AsyncMock(return_value=fast_groq_stream())

    output_chunks = []
    async for chunk in service.generate_conversational_stream("What is RAG?"):
        output_chunks.append(chunk)

    full_response = "".join(output_chunks)
    assert "Answer from Groq fallback" in full_response
    assert service.hf_exhausted is True
