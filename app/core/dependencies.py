import os
import uuid
from typing import Optional
from functools import lru_cache
from fastapi import Header, HTTPException, status
from services.document_service import DocumentService
from services.llm_service import LLMService
from core.rag_pipeline import VectorlessRAG
from core.storage import storage, MemoryStorage


DEFAULT_CLIENT_ID = "00000000-0000-4000-8000-000000000000"


def get_client_id(x_client_id: Optional[str] = Header(None, alias="X-Client-ID")) -> str:
    """FastAPI dependency validating the X-Client-ID header as a valid UUIDv4.
    If missing or invalid, gracefully defaults to DEFAULT_CLIENT_ID to ensure
    resilience against missing headers, health checks, or unauthenticated clients.
    """
    if not x_client_id:
        return DEFAULT_CLIENT_ID
    try:
        val = uuid.UUID(x_client_id.strip(), version=4)
        return str(val)
    except (ValueError, AttributeError):
        try:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, x_client_id.strip()))
        except Exception:
            return DEFAULT_CLIENT_ID


@lru_cache()
def get_document_service() -> DocumentService:
    return DocumentService(api_key=os.getenv("PAGEINDEX_API_KEY"))


@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache()
def get_rag_pipeline() -> VectorlessRAG:
    return VectorlessRAG(
        get_llm_service(),
        memory_service=get_memory_service(),
        storage_instance=get_storage(),
    )


def get_storage() -> MemoryStorage:
    return storage


@lru_cache()
def get_memory_service() -> 'MemoryService':
    from services.memory_service import MemoryService
    return MemoryService(get_llm_service())


from core.rate_limiter import DualKeyRateLimitChecker

rate_limit_ask = DualKeyRateLimitChecker("ask", default_limit=20, env_var_name="RATE_LIMIT_ASK_PER_MIN")
rate_limit_upload = DualKeyRateLimitChecker("upload", default_limit=5, env_var_name="RATE_LIMIT_UPLOAD_PER_MIN")
rate_limit_quiz = DualKeyRateLimitChecker("quiz", default_limit=5, env_var_name="RATE_LIMIT_QUIZ_PER_MIN")
rate_limit_tools = DualKeyRateLimitChecker("tools", default_limit=20, env_var_name="RATE_LIMIT_TOOLS_PER_MIN")


