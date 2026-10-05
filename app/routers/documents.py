import os
import time
import uuid
import asyncio
import hashlib
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Response, status
from services.document_service import DocumentService
from services.llm_service import LLMService
from core.storage import MemoryStorage
from core.dependencies import get_document_service, get_llm_service, get_storage
from schemas.schemas import UploadResponse, DocumentStatusResponse
from utils.logger import app_logger

router = APIRouter(tags=["Documents"])


async def run_background_indexing(
    doc_id: str,
    file_path: str,
    filename: str,
    content_hash: str,
    doc_service: DocumentService,
    llm_service: LLMService,
    storage: MemoryStorage,
):
    """Background task for long-running PDF parsing, PageIndex polling, and fallback extraction."""
    t_start = time.monotonic()
    try:
        def on_stage(stage: str, progress: int):
            storage.update_document_progress(
                doc_id=doc_id,
                status="indexing",
                current_stage=stage,
                progress=progress,
            )

        # 1. Execute document parsing in worker thread (blocking network/fitz ops)
        res = await asyncio.to_thread(
            doc_service.process_pdf,
            file_path=file_path,
            document_id=doc_id,
            on_stage=on_stage,
        )

        tree = res["tree"]
        page_count = res["page_count"]
        indexing_method = res["indexing_method"]
        failure_reason = res.get("failure_reason")

        # 2. Validation & index normalization
        app_logger.info(f"[INDEX] document_id={doc_id} stage=validation")
        on_stage("Validating index", 90)

        tree = llm_service.normalize_tree(tree)
        node_map, flat_sections = llm_service.build_index_maps(tree)
        section_count = len(flat_sections)

        # 3. Store in-memory acceleration indexes
        storage.set_node_map(doc_id, node_map)
        storage.set_flat_sections(doc_id, flat_sections)

        total_duration = round(time.monotonic() - t_start, 2)

        # 4. Mark document ready and persist
        storage.set_document_ready(
            doc_id=doc_id,
            tree=tree,
            page_count=page_count,
            section_count=section_count,
            indexing_method=indexing_method,
            duration=total_duration,
            filename=filename,
            file_path=file_path,
        )
        storage.set_hash_mapping(content_hash, doc_id)

        app_logger.info(
            f"[INDEX] document_id={doc_id} stage=ready duration={total_duration:.2f}s "
            f"method={indexing_method} pages={page_count} sections={section_count}"
        )

    except Exception as e:
        total_duration = round(time.monotonic() - t_start, 2)
        err_msg = str(e)
        app_logger.error(
            f"[INDEX] document_id={doc_id} stage=failed error='{err_msg}' duration={total_duration:.2f}s",
            exc_info=True,
        )
        storage.set_document_error(doc_id, err_msg, duration=total_duration)
    finally:
        storage.unregister_active_task(doc_id)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(
    response: Response,
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    doc_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
    storage: MemoryStorage = Depends(get_storage),
):
    """
    Asynchronous PDF upload and indexing endpoint.
    Validates PDF, checks deduplication by SHA-256 hash, starts background indexing job,
    and returns HTTP 202 Accepted immediately.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF documents are supported."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF file is empty."
        )

    content_hash = hashlib.sha256(contents).hexdigest()

    # Deduplication check: Check if same document is already ready or currently indexing
    existing_doc = storage.get_doc_by_hash(content_hash)
    if existing_doc:
        doc_id = existing_doc["doc_id"]
        doc_status = existing_doc.get("status")

        if chat_id:
            storage.set_chat_doc(chat_id, doc_id)

        if doc_status == "ready":
            app_logger.info(
                f"[INDEX] document_id={doc_id} stage=ready reuse=True cache_hit=True "
                f"pages={existing_doc.get('page_count', 1)} sections={existing_doc.get('section_count', 0)}"
            )
            response.status_code = status.HTTP_200_OK
            return UploadResponse(
                message="Success (reused cached index)",
                doc_id=doc_id,
                document_id=doc_id,
                filename=existing_doc.get("filename", file.filename),
                status="ready",
                page_count=existing_doc.get("page_count", 1),
            )

        if doc_status == "indexing":
            app_logger.info(
                f"[INDEX] document_id={doc_id} stage=indexing duplicate_ignored=True"
            )
            response.status_code = status.HTTP_202_ACCEPTED
            return UploadResponse(
                message="Document is already being indexed",
                doc_id=doc_id,
                document_id=doc_id,
                filename=existing_doc.get("filename", file.filename),
                status="indexing",
                page_count=0,
            )

    # Persist file to local disk
    doc_id = str(uuid.uuid4())
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    app_logger.info(
        f"[INDEX] document_id={doc_id} stage=upload filename='{file.filename}' "
        f"size={len(contents)} hash={content_hash[:12]}"
    )

    # Register document in storage
    storage.create_document(
        doc_id=doc_id,
        filename=file.filename,
        status="indexing",
        content_hash=content_hash,
        file_path=file_path,
        current_stage="Processing PDF",
        progress=10,
    )

    if chat_id:
        storage.set_chat_doc(chat_id, doc_id)
        app_logger.info(f"Associated document {doc_id} ('{file.filename}') with chat {chat_id}")

    # Launch detached asynchronous background task
    task = asyncio.create_task(
        run_background_indexing(
            doc_id=doc_id,
            file_path=file_path,
            filename=file.filename,
            content_hash=content_hash,
            doc_service=doc_service,
            llm_service=llm_service,
            storage=storage,
        )
    )
    storage.register_active_task(doc_id, task)

    response.status_code = status.HTTP_202_ACCEPTED
    return UploadResponse(
        message="Indexing started",
        doc_id=doc_id,
        document_id=doc_id,
        filename=file.filename,
        status="indexing",
        page_count=0,
    )


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    storage: MemoryStorage = Depends(get_storage),
):
    """Retrieve indexing/ready/failed status and operational metadata for a document."""
    doc = storage.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document ID not found.")

    doc_status = doc.get("status", "unknown")
    progress = doc.get("progress", 0)
    current_stage = doc.get("current_stage", "uploaded")

    if doc_status == "ready":
        progress = 100
        current_stage = "Ready"
    elif doc_status in ("failed", "error"):
        doc_status = "failed"
        current_stage = "Failed"

    return DocumentStatusResponse(
        document_id=doc.get("doc_id", document_id),
        doc_id=doc.get("doc_id", document_id),
        filename=doc.get("filename", "document.pdf"),
        status=doc_status,
        progress=progress,
        current_stage=current_stage,
        page_count=doc.get("page_count", 0),
        section_count=doc.get("section_count", 0),
        indexing_method=doc.get("indexing_method"),
        duration=doc.get("duration"),
        error=doc.get("error"),
    )


@router.post("/documents/{document_id}/retry", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def retry_document_indexing(
    document_id: str,
    doc_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
    storage: MemoryStorage = Depends(get_storage),
):
    """Explicitly retry indexing for a failed document."""
    doc = storage.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document ID not found.")

    if storage.is_task_active(document_id) or doc.get("status") == "indexing":
        return UploadResponse(
            message="Document is already indexing",
            doc_id=document_id,
            document_id=document_id,
            filename=doc.get("filename"),
            status="indexing",
            page_count=0,
        )

    file_path = doc.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail="Document file no longer exists on disk. Please upload again."
        )

    # Reset status
    storage.update_document_progress(
        doc_id=document_id,
        status="indexing",
        current_stage="Processing PDF",
        progress=10,
    )

    task = asyncio.create_task(
        run_background_indexing(
            doc_id=document_id,
            file_path=file_path,
            filename=doc.get("filename", "document.pdf"),
            content_hash=doc.get("content_hash", ""),
            doc_service=doc_service,
            llm_service=llm_service,
            storage=storage,
        )
    )
    storage.register_active_task(document_id, task)

    return UploadResponse(
        message="Re-indexing started",
        doc_id=document_id,
        document_id=document_id,
        filename=doc.get("filename"),
        status="indexing",
        page_count=0,
    )
