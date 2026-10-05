import os
from functools import lru_cache
from services.document_service import DocumentService
from services.llm_service import LLMService
from core.rag_pipeline import VectorlessRAG
from core.storage import storage, MemoryStorage


@lru_cache()
def get_document_service() -> DocumentService:
    return DocumentService(api_key=os.getenv("PAGEINDEX_API_KEY"))


@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache()
def get_rag_pipeline() -> VectorlessRAG:
    return VectorlessRAG(get_llm_service())


def get_storage() -> MemoryStorage:
    return storage
