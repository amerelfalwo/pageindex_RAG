import os
import time
import asyncio
import hashlib
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from services.document_service import DocumentService
from services.llm_service import LLMService
from core.storage import MemoryStorage
from core.dependencies import get_document_service, get_llm_service, get_storage
from schemas.schemas import UploadResponse, DocumentStatusResponse
from utils.logger import app_logger

router = APIRouter(tags=["Documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    doc_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
    storage: MemoryStorage = Depends(get_storage),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF documents are supported."
        )

    t_upload_start = time.monotonic()
    contents = await file.read()
    content_hash = hashlib.sha256(contents).hexdigest()

    # Check content-hash cache: Instant reuse if document was already processed
    existing_doc = storage.get_doc_by_hash(content_hash)
    if existing_doc:
        doc_id = existing_doc["doc_id"]
        if chat_id:
            storage.set_chat_doc(chat_id, doc_id)
        duration = time.monotonic() - t_upload_start
        app_logger.info(
            f"[PERF] upload_document doc_id={doc_id} filename='{file.filename}' "
            f"cache_hit=True duration={duration:.3f}s pages={existing_doc.get('page_count', 1)}"
        )
        return UploadResponse(
            message="Success (reused cached index)",
            doc_id=doc_id,
            filename=file.filename,
            status="ready",
            page_count=existing_doc.get("page_count", 1),
        )

    # Pre-register new document in indexing state
    doc_id = storage.create_document(filename=file.filename, status="indexing", content_hash=content_hash)
    if chat_id:
        storage.set_chat_doc(chat_id, doc_id)
        app_logger.info(f"Associated document {doc_id} ('{file.filename}') with chat {chat_id} in indexing state.")

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    file_path = os.path.join("data", f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    try:
        app_logger.info(f"Processing uploaded PDF: {file.filename} (doc_id: {doc_id})")
        # Run synchronous PDF parsing / PageIndex polling in a worker thread so the event loop is never blocked
        tree = await asyncio.to_thread(doc_service.process_pdf, file_path)
        tree = llm_service.normalize_tree(tree)
        page_count = len(tree) if isinstance(tree, list) else 1

        # Pre-compute fast lookup tables for instant retrieval
        node_map, flat_sections = llm_service.build_index_maps(tree)
        storage.set_node_map(doc_id, node_map)
        storage.set_flat_sections(doc_id, flat_sections)

        storage.set_document_ready(
            doc_id=doc_id,
            tree=tree,
            page_count=page_count,
            filename=file.filename,
        )
        storage.set_hash_mapping(content_hash, doc_id)

        duration = time.monotonic() - t_upload_start
        app_logger.info(
            f"[PERF] upload_document doc_id={doc_id} filename='{file.filename}' "
            f"cache_hit=False duration={duration:.2f}s pages={page_count} sections={len(flat_sections)}"
        )
        return UploadResponse(
            message="Success",
            doc_id=doc_id,
            filename=file.filename,
            status="ready",
            page_count=page_count,
        )
    except Exception as e:
        storage.set_document_error(doc_id, str(e))
        app_logger.error(f"Failed to process PDF {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # We no longer remove the file as requested by the user to keep it in the data folder
        pass

@router.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Retrieve indexing/ready status and metadata for a document."""
    doc = storage.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document ID not found.")
    return DocumentStatusResponse(
        doc_id=doc["doc_id"],
        filename=doc.get("filename", "document.pdf"),
        status=doc.get("status", "unknown"),
        page_count=doc.get("page_count", 0),
        error=doc.get("error"),
    )

