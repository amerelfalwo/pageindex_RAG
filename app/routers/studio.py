import os
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from core.dependencies import (
    get_storage,
    get_document_service,
    get_llm_service,
    get_memory_service,
    get_client_id,
    rate_limit_quiz,
)
from core.storage import MemoryStorage
from services.document_service import DocumentService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from utils.logger import app_logger

router = APIRouter(prefix="/studio", tags=["Study Studio"])


class QuizGenerateRequest(BaseModel):
    num_mcqs: Optional[int] = 5
    num_flashcards: Optional[int] = 5
    language: Optional[str] = "auto"


@router.get("/{doc_id}/mindmap")
async def get_document_mindmap(
    doc_id: str,
    client_id: str = Depends(get_client_id),
    storage: MemoryStorage = Depends(get_storage),
    doc_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    """
    Transforms the hierarchical section tree and knowledge dossier of a document
    into a structured node-edge graph for interactive visualization.
    Each node includes: id, label, section_title, page_number, type, summary, level.
    """
    doc = storage.get_document(doc_id, owner_id=client_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document ID not found.")

    filename = doc.get("filename", "Document.pdf")
    tree = storage.get_document_tree(doc_id)
    if tree is None and doc.get("file_path") and os.path.exists(doc["file_path"]):
        tree = doc_service._local_pdf_parse(doc["file_path"])
        tree = llm_service.normalize_tree(tree)
        node_map, flat_sections = llm_service.build_index_maps(tree)
        storage.set_document_tree(doc_id, tree)
        storage.set_node_map(doc_id, node_map)
        storage.set_flat_sections(doc_id, flat_sections)

    if not tree:
        tree = []

    # Retrieve dossier if available
    dossier = storage.get_knowledge_dossier(doc_id, owner_id=client_id) or {}
    topics = dossier.get("topics") or []
    key_findings = dossier.get("key_findings") or []
    node_map = storage.get_node_map(doc_id) or {}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # 1. Central Document Root Node (Level 0)
    root_node_id = f"doc_{doc_id}"
    root_summary = dossier.get("executive_summary") or f"Knowledge graph for {filename}"
    nodes.append({
        "id": root_node_id,
        "label": filename,
        "section_title": filename,
        "page_number": 1,
        "type": "root",
        "summary": root_summary,
        "text": root_summary,
        "level": 0,
    })

    # 2. Executive Dossier Hub (Level 1)
    if topics or key_findings:
        dossier_hub_id = f"dossier_{doc_id}"
        dossier_text = "\n".join([f"• {f}" for f in key_findings]) if key_findings else f"{len(topics)} topics identified."
        nodes.append({
            "id": dossier_hub_id,
            "label": "Core Themes & Findings",
            "section_title": "Executive Knowledge Dossier",
            "page_number": 1,
            "type": "dossier",
            "summary": f"{len(topics)} topics & {len(key_findings)} core findings identified.",
            "text": dossier_text,
            "level": 1,
        })
        edges.append({
            "id": f"edge_{root_node_id}_{dossier_hub_id}",
            "source": root_node_id,
            "target": dossier_hub_id,
            "label": "synthesizes",
        })

        # Connect individual topics/findings (Level 2)
        for i, topic in enumerate(topics[:6]):
            topic_node_id = f"topic_{doc_id}_{i}"
            nodes.append({
                "id": topic_node_id,
                "label": topic,
                "section_title": f"Topic: {topic}",
                "page_number": 1,
                "type": "topic",
                "summary": f"Key concept identified in {filename}",
                "text": f"Topic '{topic}' extracted as a key thematic finding from {filename}.",
                "level": 2,
            })
            edges.append({
                "id": f"edge_{dossier_hub_id}_{topic_node_id}",
                "source": dossier_hub_id,
                "target": topic_node_id,
                "label": "topic",
            })

    # 3. Document Hierarchical Outline (Chapters & Sections)
    def add_tree_nodes(items, parent_id, level):
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            nid = item.get("node_id") or f"node_{level}_{idx}_{parent_id}"
            title = item.get("title") or item.get("name") or f"Section {idx+1}"
            page = item.get("page_index") or item.get("page") or 1

            node_map_entry = node_map.get(nid, {}) if isinstance(node_map, dict) else {}
            node_text = (
                item.get("text")
                or item.get("content")
                or node_map_entry.get("text")
                or node_map_entry.get("content")
                or ""
            ).strip()
            summary = item.get("summary") or node_map_entry.get("summary") or node_text[:200].strip() or f"Section {title}"

            node_type = "chapter" if level == 1 else "section"
            nodes.append({
                "id": nid,
                "label": title,
                "section_title": title,
                "page_number": page,
                "type": node_type,
                "summary": summary,
                "text": node_text or summary,
                "level": level,
            })
            edges.append({
                "id": f"edge_{parent_id}_{nid}",
                "source": parent_id,
                "target": nid,
                "label": "contains" if level == 1 else "sub-section",
            })

            # Traverse child sub-sections (limit to max 3 levels deep for graph clarity)
            child_nodes = item.get("nodes")
            if child_nodes and isinstance(child_nodes, list) and level < 3:
                add_tree_nodes(child_nodes, nid, level + 1)

    add_tree_nodes(tree, root_node_id, level=1)

    app_logger.info(
        f"[STUDIO-MINDMAP] Generated graph for doc_id={doc_id} with "
        f"{len(nodes)} nodes and {len(edges)} edges."
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "doc_id": doc_id,
            "filename": filename,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "generated_at": time.time(),
        },
    }


@router.post("/{doc_id}/quiz")
async def generate_document_quiz(
    doc_id: str,
    payload: Optional[QuizGenerateRequest] = None,
    client_id: str = Depends(get_client_id),
    storage: MemoryStorage = Depends(get_storage),
    doc_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
    _rl: None = Depends(rate_limit_quiz),
):
    """
    Generates 5 Multiple Choice Questions and 5 Study Flashcards for a document
    using LLMService with HF + Groq fallback and ground truth tree/dossier context.
    Each item includes question, options, answer, explanation, and exact citation (Section, Page).
    """
    t0 = time.monotonic()
    doc = storage.get_document(doc_id, owner_id=client_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document ID not found.")

    filename = doc.get("filename", "Document.pdf")
    tree = storage.get_document_tree(doc_id)
    if tree is None and doc.get("file_path") and os.path.exists(doc["file_path"]):
        tree = doc_service._local_pdf_parse(doc["file_path"])
        tree = llm_service.normalize_tree(tree)
        storage.set_document_tree(doc_id, tree)

    dossier = storage.get_knowledge_dossier(doc_id, owner_id=client_id)

    app_logger.info(f"[STUDIO-QUIZ] Generating quiz and flashcards for doc_id={doc_id} ('{filename}')...")
    result = await llm_service.generate_quiz_and_flashcards(
        filename=filename,
        tree=tree or [],
        dossier=dossier,
    )

    elapsed = time.monotonic() - t0
    app_logger.info(
        f"[STUDIO-QUIZ] Generated {len(result['mcqs'])} MCQs and "
        f"{len(result['flashcards'])} Flashcards in {elapsed:.2f}s."
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "mcqs": result["mcqs"],
        "flashcards": result["flashcards"],
        "generated_at": time.time(),
        "duration_seconds": round(elapsed, 2),
    }
