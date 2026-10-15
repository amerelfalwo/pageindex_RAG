import time
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from core.dependencies import get_memory_service, get_storage
from core.storage import MemoryStorage
from services.memory_service import MemoryService
from utils.logger import app_logger

router = APIRouter(prefix="/memory", tags=["Memory & Knowledge"])


@router.get("/knowledge")
async def get_global_knowledge(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Retrieve workspace collective knowledge summary and all per-document knowledge dossiers.
    Provides cross-document intelligence and long-term memory across sessions.
    """
    return memory_service.get_global_knowledge()


@router.post("/knowledge/refresh")
async def refresh_global_knowledge(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Synthesize and rebuild the global workspace knowledge memory across all indexed documents.
    """
    app_logger.info("Triggered manual refresh of global knowledge memory")
    result = await memory_service.synthesize_global_knowledge()
    dossiers = memory_service.list_knowledge_dossiers()
    return {
        "message": "Global knowledge memory refreshed successfully",
        "collective_summary": result.get("collective_summary"),
        "key_insights": result.get("key_insights", []),
        "total_documents": result.get("total_documents", len(dossiers)),
        "dossiers": dossiers,
    }


@router.get("/dossier/{doc_id}")
async def get_document_dossier(
    doc_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Retrieve structured knowledge dossier for an individual document.
    """
    dossier = memory_service.get_document_dossier(doc_id)
    if not dossier:
        raise HTTPException(status_code=404, detail="Knowledge dossier not found for this document")
    return dossier


@router.get("/chat/{chat_id}")
async def get_chat_memory_context(
    chat_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
    storage: MemoryStorage = Depends(get_storage),
):
    """
    Inspect the short-term and long-term isolated memory for a specific chat session.
    """
    if not storage.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    context = memory_service.get_session_context(chat_id)
    attached_docs = storage.get_chat_docs(chat_id)
    return {
        "chat_id": chat_id,
        "session_summary": context.get("session_summary"),
        "recent_turns_count": len(context.get("recent_turns", [])),
        "recent_turns": context.get("recent_turns", []),
        "attached_documents": attached_docs,
    }


@router.post("/chat/{chat_id}/summarize")
async def force_summarize_chat(
    chat_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
    storage: MemoryStorage = Depends(get_storage),
):
    """
    Force-generate or update the rolling long-term memory summary for this chat session.
    """
    if not storage.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    summary = await memory_service.update_session_summary_if_needed(chat_id, force=True)
    return {
        "message": "Chat session summary updated",
        "chat_id": chat_id,
        "summary": summary,
    }
