import asyncio
import re
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from schemas.schemas import (
    AskRequest,
    NewChatResponse,
    ChatHistoryResponse,
    AttachDocRequest,
    AttachMultipleDocsRequest,
)
from core.storage import MemoryStorage
from core.dependencies import (
    get_llm_service,
    get_rag_pipeline,
    get_storage,
    get_memory_service,
)
from core.rag_pipeline import VectorlessRAG
from services.llm_service import LLMService
from services.memory_service import MemoryService
from utils.logger import app_logger

router = APIRouter(tags=["Chat"])


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
    """Attach an indexed document to an active conversation session."""
    if not storage.tree_exists(body.doc_id):
        raise HTTPException(status_code=404, detail="Document ID not found in memory")
    storage.add_chat_doc(chat_id, body.doc_id, doc_name=body.doc_name)
    app_logger.info(f"Attached document {body.doc_id} to chat session {chat_id}")
    return {"message": "Document attached successfully", "chat_id": chat_id, "doc_id": body.doc_id}


@router.post("/chat/{chat_id}/attach-multiple")
async def attach_multiple_documents_to_chat(
    chat_id: str,
    body: AttachMultipleDocsRequest,
    storage: MemoryStorage = Depends(get_storage),
):
    """Attach multiple indexed documents to an active conversation session."""
    attached = []
    for doc_id in body.doc_ids:
        if storage.tree_exists(doc_id):
            storage.add_chat_doc(chat_id, doc_id)
            attached.append(doc_id)
    app_logger.info(f"Attached {len(attached)} documents to chat session {chat_id}")
    return {"message": f"Attached {len(attached)} documents successfully", "chat_id": chat_id, "doc_ids": attached}


@router.get("/chat/{chat_id}/documents")
async def get_chat_documents(
    chat_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Return all attached documents for this session with status and page counts."""
    docs = storage.get_chat_docs(chat_id)
    detailed_docs = []
    for d in docs:
        doc_info = storage.get_document(d["doc_id"])
        if doc_info:
            detailed_docs.append({
                "doc_id": d["doc_id"],
                "filename": doc_info.get("filename", d.get("doc_name", "document.pdf")),
                "status": doc_info.get("status", "ready"),
                "page_count": doc_info.get("page_count", 1),
                "section_count": doc_info.get("section_count", 0),
            })
        else:
            detailed_docs.append(d)
    return {"chat_id": chat_id, "documents": detailed_docs}


@router.delete("/chat/{chat_id}/document/{doc_id}")
async def detach_specific_document_from_chat(
    chat_id: str,
    doc_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Detach a specific document from the active chat session."""
    storage.remove_chat_doc(chat_id, doc_id=doc_id)
    app_logger.info(f"Detached document {doc_id} from chat session {chat_id}")
    return {"message": "Document detached successfully", "chat_id": chat_id, "doc_id": doc_id}


@router.delete("/chat/{chat_id}/document")
async def detach_document_from_chat(
    chat_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Detach all documents from active chat session."""
    storage.clear_chat_docs(chat_id)
    storage.remove_chat_doc(chat_id)
    app_logger.info(f"Cleared all documents from chat session {chat_id}")
    return {"message": "All documents detached successfully", "chat_id": chat_id}


@router.post("/ask")
async def ask_question(
    request: AskRequest,
    storage: MemoryStorage = Depends(get_storage),
    llm_service: LLMService = Depends(get_llm_service),
    rag_pipeline: VectorlessRAG = Depends(get_rag_pipeline),
    memory_service: MemoryService = Depends(get_memory_service),
):
    # Ensure chat session exists in persistent storage
    storage.ensure_chat(request.chat_id)

    # 1. Resolve target document IDs for this query session
    target_doc_ids: List[str] = []
    if request.doc_ids:
        target_doc_ids.extend(request.doc_ids)
    elif request.doc_id:
        target_doc_ids.append(request.doc_id)

    if target_doc_ids:
        # Associate newly requested docs with this session
        for did in target_doc_ids:
            storage.add_chat_doc(request.chat_id, did)
    else:
        # Check existing docs attached to this session
        session_docs = storage.get_chat_docs(request.chat_id)
        if session_docs:
            target_doc_ids = [d["doc_id"] for d in session_docs]
        else:
            primary_id = storage.get_chat_doc(request.chat_id)
            if primary_id:
                target_doc_ids = [primary_id]

    # Deduplicate while preserving order
    target_doc_ids = list(dict.fromkeys(target_doc_ids))

    # 2. Inspect document states
    ready_docs = []
    indexing_docs = []
    error_docs = []

    for did in target_doc_ids:
        doc_state = storage.get_document(did)
        if not doc_state:
            error_docs.append({"doc_id": did, "filename": did, "error": "Document not found or expired"})
        elif doc_state.get("status") == "ready" and doc_state.get("tree"):
            node_map = storage.get_node_map(did)
            flat_sections = storage.get_flat_sections(did)
            if node_map is None or flat_sections is None:
                node_map, flat_sections = llm_service.build_index_maps(doc_state["tree"])
                storage.set_node_map(did, node_map)
                storage.set_flat_sections(did, flat_sections)
            ready_docs.append({
                "doc_id": did,
                "filename": doc_state.get("filename", "document.pdf"),
                "tree": doc_state["tree"],
                "node_map": node_map,
                "flat_sections": flat_sections,
            })
        elif doc_state.get("status") == "indexing":
            indexing_docs.append(doc_state)
        else:
            error_docs.append(doc_state)

    # 3. Determine execution mode
    if ready_docs:
        selected_mode = "multi_doc_pageindex"
    elif indexing_docs:
        selected_mode = "indexing"
    elif error_docs and not ready_docs:
        selected_mode = "error"
    else:
        # Pure conversational AI assistant with workspace global knowledge
        selected_mode = "conversational"

    # Router Audit Logging
    app_logger.info(
        f"[CHAT ROUTER] conversation_id={request.chat_id} target_docs={len(target_doc_ids)} "
        f"ready_docs={len(ready_docs)} indexing_docs={len(indexing_docs)} "
        f"mode={selected_mode} deep_analysis={bool(request.deep_analysis)}"
    )

    # 4. Retrieve session memory (isolated short-term window + long-term rolling summary)
    mem_context = memory_service.get_session_context(request.chat_id)
    recent_turns = mem_context.get("recent_turns", [])
    session_summary = mem_context.get("session_summary")

    # 5. Retrieve global workspace knowledge summary
    global_knowledge = memory_service.get_global_knowledge()

    is_arabic = any("\u0600" <= c <= "\u06ff" for c in request.query)

    async def stream_generator():
        full_answer = ""
        try:
            if selected_mode == "multi_doc_pageindex":
                # Mode A: Deterministic Vectorless multi-doc tree retrieval & grounded synthesis
                async for chunk in rag_pipeline.answer_multi_doc_query_stream(
                    query=request.query,
                    docs=ready_docs,
                    chat_history=recent_turns,
                    session_summary=session_summary,
                    global_knowledge=global_knowledge,
                    deep_analysis=bool(request.deep_analysis),
                ):
                    full_answer += chunk
                    yield chunk

            elif selected_mode == "indexing":
                # Mode B: Document(s) attached but still indexing
                doc_names = ", ".join([d.get("filename", "PDF") for d in indexing_docs])
                if is_arabic:
                    msg = f"المستندات التالية قيد الفهرسة حالياً ({doc_names}). يرجى الانتظار لحظة حتى تكتمل المعالجة."
                else:
                    msg = f"The following documents are currently being indexed ({doc_names}). Please wait a moment."
                full_answer = msg
                yield msg

            elif selected_mode == "error":
                # Mode C: Requested documents failed or unavailable
                err_names = ", ".join([d.get("filename", "PDF") for d in error_docs])
                if is_arabic:
                    msg = f"تعذر استرجاع المستندات التالية ({err_names}). يرجى إعادة رفع الملفات."
                else:
                    msg = f"Could not access the requested documents ({err_names}). Please try re-uploading."
                full_answer = msg
                yield msg

            else:
                # Mode D: Pure conversational AI assistant with workspace collective knowledge injected
                async for chunk in llm_service.generate_conversational_stream(
                    query=request.query,
                    chat_history=recent_turns,
                    session_summary=session_summary,
                    global_knowledge=global_knowledge,
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
                # Asynchronously update long-term rolling summary if conversation threshold reached
                try:
                    asyncio.create_task(
                        memory_service.update_session_summary_if_needed(request.chat_id)
                    )
                except Exception:
                    pass

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    history = storage.get_chat_history(chat_id)
    return ChatHistoryResponse(history=history or [])


# ── Long Memory & Conversation Management Endpoints ────────────────────────

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
