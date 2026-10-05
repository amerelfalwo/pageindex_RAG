from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from schemas.schemas import AskRequest, NewChatResponse, ChatHistoryResponse
from core.storage import MemoryStorage
from core.dependencies import get_llm_service, get_rag_pipeline, get_storage
from core.rag_pipeline import VectorlessRAG
from services.llm_service import LLMService
from utils.logger import app_logger
import re

router = APIRouter(tags=["Chat"])


class AttachDocRequest(BaseModel):
    doc_id: str


@router.post("/chat/new", response_model=NewChatResponse)
async def create_chat(storage: MemoryStorage = Depends(get_storage)):
    chat_id = storage.create_chat()
    app_logger.info(f"Created new chat session: {chat_id}")
    return NewChatResponse(chat_id=chat_id)


@router.post("/chat/{chat_id}/attach")
async def attach_document_to_chat(
    chat_id: str,
    body: AttachDocRequest,
    storage: MemoryStorage = Depends(get_storage),
):
    """Attach an indexed document to an active conversation."""
    if not storage.tree_exists(body.doc_id):
        raise HTTPException(status_code=404, detail="Document ID not found in memory")
    storage.set_chat_doc(chat_id, body.doc_id)
    app_logger.info(f"Attached document {body.doc_id} to chat session {chat_id}")
    return {"message": "Document attached successfully", "chat_id": chat_id, "doc_id": body.doc_id}


@router.delete("/chat/{chat_id}/document")
async def detach_document_from_chat(
    chat_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Detach document from active chat session."""
    storage.remove_chat_doc(chat_id)
    app_logger.info(f"Detached document from chat session {chat_id}")
    return {"message": "Document detached successfully", "chat_id": chat_id}


@router.post("/ask")
async def ask_question(
    request: AskRequest,
    storage: MemoryStorage = Depends(get_storage),
    llm_service: LLMService = Depends(get_llm_service),
    rag_pipeline: VectorlessRAG = Depends(get_rag_pipeline),
):
    chat_history = storage.ensure_chat(request.chat_id)

    # 1. Resolve active document identity for this chat session
    effective_doc_id = request.doc_id or storage.get_chat_doc(request.chat_id)
    doc_state = storage.get_document(effective_doc_id) if effective_doc_id else None

    # If request passed doc_id, ensure session association is persisted
    if request.doc_id and effective_doc_id:
        storage.set_chat_doc(request.chat_id, effective_doc_id)

    # 2. Deterministic routing decision
    error_detail = None
    if not effective_doc_id:
        selected_mode = "conversational"
    elif doc_state is None:
        selected_mode = "error"
        error_detail = "Document session not found or expired"
    elif doc_state.get("status") == "indexing":
        selected_mode = "indexing"
    elif doc_state.get("status") == "ready" and doc_state.get("tree"):
        selected_mode = "pageindex"
    else:
        selected_mode = "error"
        error_detail = doc_state.get("error") or "Document tree unavailable"

    # Router Audit Logging
    app_logger.info(
        f"[CHAT ROUTER] conversation_id={request.chat_id} document_id={effective_doc_id} "
        f"document_status={doc_state.get('status') if doc_state else 'none'} "
        f"tree_ready={bool(doc_state and doc_state.get('tree'))} mode={selected_mode} "
        f"deep_analysis={bool(request.deep_analysis)}"
    )

    # Use concise recent history directly to preserve TTFT without blocking LLM calls
    effective_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

    is_arabic = any("\u0600" <= c <= "\u06ff" for c in request.query)
    doc_name = (doc_state.get("filename") if doc_state else None) or request.doc_name

    async def stream_generator():
        full_answer = ""
        try:
            if selected_mode == "pageindex":
                # Mode A: Deterministic PageIndex tree retrieval & grounded synthesis
                tree = doc_state["tree"]
                node_map = storage.get_node_map(effective_doc_id)
                flat_sections = storage.get_flat_sections(effective_doc_id)
                if node_map is None or flat_sections is None:
                    node_map, flat_sections = llm_service.build_index_maps(tree)
                    storage.set_node_map(effective_doc_id, node_map)
                    storage.set_flat_sections(effective_doc_id, flat_sections)

                async for chunk in rag_pipeline.answer_query_stream(
                    request.query,
                    tree,
                    chat_history=effective_history,
                    doc_name=doc_name,
                    node_map=node_map,
                    flat_sections=flat_sections,
                    deep_analysis=bool(request.deep_analysis),
                ):
                    full_answer += chunk
                    yield chunk

            elif selected_mode == "indexing":
                # Mode B: Document attached but still indexing - inform user explicitly
                filename_display = doc_name or "PDF"
                if is_arabic:
                    msg = f"المستند ({filename_display}) قيد المعالجة والفهرسة حالياً. يرجى الانتظار لحظة حتى تكتمل الفهرسة."
                else:
                    msg = f"Your document ({filename_display}) is currently being indexed. Please wait a moment."
                full_answer = msg
                yield msg

            elif selected_mode == "error":
                # Mode C: Document failed or unavailable - do NOT fall back to conversational
                filename_display = doc_name or "PDF"
                detail_str = error_detail or "File unavailable"
                if is_arabic:
                    msg = f"تعذر الوصول إلى المستند المطلوب ({detail_str}). يرجى إعادة رفع الملف."
                else:
                    msg = f"Could not access the requested document ({detail_str}). Please try re-uploading the file."
                full_answer = msg
                yield msg

            else:
                # Mode D: Pure conversational AI assistant (strictly when no document is attached)
                async for chunk in llm_service.generate_conversational_stream(
                    request.query, chat_history
                ):
                    full_answer += chunk
                    yield chunk

        finally:
            if full_answer:
                clean_answer = re.sub(r'<!--STAGE:.*?-->\n?', '', full_answer).strip()
                storage.add_message(request.chat_id, role="user", content=request.query)
                storage.add_message(
                    request.chat_id, role="assistant", content=clean_answer or full_answer
                )

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    history = storage.get_chat_history(chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Chat ID not found")
    return ChatHistoryResponse(history=history)


# ── Long Memory Endpoints ─────────────────────────────────────────────────

@router.get("/chats")
async def list_all_chats(storage: MemoryStorage = Depends(get_storage)):
    """Return all persisted chat sessions ordered by most recent."""
    chats = storage.list_chats()
    return {"chats": chats}


@router.delete("/chat/{chat_id}")
async def delete_chat(
    chat_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Permanently delete a chat session and all its messages."""
    storage.delete_chat(chat_id)
    app_logger.info(f"Deleted chat session: {chat_id}")
    return {"message": "Chat deleted", "chat_id": chat_id}


class RenameChatRequest(BaseModel):
    title: str


@router.patch("/chat/{chat_id}/title")
async def rename_chat(
    chat_id: str,
    body: RenameChatRequest,
    storage: MemoryStorage = Depends(get_storage),
):
    """Update the display title of a chat session."""
    storage.update_chat_title(chat_id, body.title)
    return {"message": "Title updated", "chat_id": chat_id, "title": body.title}
