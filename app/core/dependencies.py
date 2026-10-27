import os
import uuid
from typing import Optional
from functools import lru_cache
from fastapi import Header, HTTPException, status
from services.document_service import DocumentService
from services.llm_service import LLMService
from core.rag_pipeline import VectorlessRAG
from core.storage import storage, MemoryStorage


def get_client_id(x_client_id: Optional[str] = Header(None, alias="X-Client-ID")) -> str:
    """FastAPI dependency validating the X-Client-ID header as a valid UUIDv4."""
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required header: X-Client-ID",
        )
    try:
        val = uuid.UUID(x_client_id.strip(), version=4)
        return str(val)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid X-Client-ID format: must be a valid UUIDv4 string",
        )


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

