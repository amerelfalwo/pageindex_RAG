import os
import json
import time
import uuid
from typing import Dict, List, Optional, Any

from core.persistent_db import persistent_db

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "index_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class MemoryStorage:
    """
    Hybrid storage:
      - In-memory for heavy objects (document trees, node_maps) for speed.
      - SQLite (via persistent_db) for chat history, chat metadata, and doc metadata.
      - Disk JSON cache for document trees (already existed).
    """

    def __init__(self) -> None:
        self._trees: Dict[str, Any] = {}
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._node_maps: Dict[str, Dict[str, Any]] = {}
        self._flat_sections: Dict[str, List[Dict[str, Any]]] = {}
        # hash → doc_id quick lookup (in-memory only; rebuilt from DB on demand)
        self._hash_to_doc_id: Dict[str, str] = {}
        # active background indexing tasks
        self._active_tasks: Dict[str, Any] = {}

    def register_active_task(self, doc_id: str, task: Any) -> None:
        self._active_tasks[doc_id] = task

    def unregister_active_task(self, doc_id: str) -> None:
        self._active_tasks.pop(doc_id, None)

    def is_task_active(self, doc_id: str) -> bool:
        task = self._active_tasks.get(doc_id)
        if task is None:
            return False
        if hasattr(task, "done"):
            return not task.done()
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Document hash / cache helpers
    # ──────────────────────────────────────────────────────────────────────
    def get_doc_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        # 1. In-memory hot cache
        doc_id = self._hash_to_doc_id.get(content_hash)
        if doc_id and doc_id in self._documents:
            doc = self._documents[doc_id]
            if doc.get("status") == "ready" and doc.get("tree"):
                return doc
            if doc.get("status") == "indexing":
                return doc

        # 2. SQLite metadata
        db_meta = persistent_db.find_doc_by_hash(content_hash)
        if db_meta:
            doc_id = db_meta["doc_id"]
            if db_meta.get("status") == "indexing":
                doc_obj = {
                    "doc_id": doc_id,
                    "document_id": doc_id,
                    "filename": db_meta.get("filename", "document.pdf"),
                    "status": "indexing",
                    "progress": db_meta.get("progress", 10),
                    "current_stage": db_meta.get("current_stage", "Processing PDF"),
                    "page_count": 0,
                    "section_count": 0,
                    "content_hash": content_hash,
                    "file_path": db_meta.get("file_path"),
                    "error": None,
                }
                self._documents[doc_id] = doc_obj
                self._hash_to_doc_id[content_hash] = doc_id
                return doc_obj

            # 3. Try to load tree from disk JSON cache for ready documents
            cache_path = os.path.join(CACHE_DIR, f"{content_hash}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    self._documents[doc_id] = {
                        "doc_id": doc_id,
                        "document_id": doc_id,
                        "filename": cached.get("filename", db_meta.get("filename", "document.pdf")),
                        "status": "ready",
                        "progress": 100,
                        "current_stage": "Ready",
                        "tree": cached.get("tree"),
                        "page_count": cached.get("page_count", db_meta.get("page_count", 1)),
                        "section_count": cached.get("section_count", db_meta.get("section_count", len(cached.get("flat_sections", [])))),
                        "indexing_method": db_meta.get("indexing_method", "pageindex"),
                        "duration": db_meta.get("duration"),
                        "error": None,
                        "content_hash": content_hash,
                        "created_at": cached.get("created_at", time.time()),
                        "updated_at": time.time(),
                    }
                    self._trees[doc_id] = cached.get("tree")
                    if cached.get("node_map"):
                        self._node_maps[doc_id] = cached["node_map"]
                    if cached.get("flat_sections"):
                        self._flat_sections[doc_id] = cached["flat_sections"]
                    self._hash_to_doc_id[content_hash] = doc_id
                    return self._documents[doc_id]
                except Exception:
                    pass
        return None

    def set_hash_mapping(self, content_hash: str, doc_id: str) -> None:
        self._hash_to_doc_id[content_hash] = doc_id
        doc = self._documents.get(doc_id)
        if doc and doc.get("status") == "ready" and doc.get("tree"):
            cache_path = os.path.join(CACHE_DIR, f"{content_hash}.json")
            try:
                payload = {
                    "doc_id": doc_id,
                    "filename": doc.get("filename"),
                    "tree": doc.get("tree"),
                    "page_count": doc.get("page_count"),
                    "node_map": self._node_maps.get(doc_id),
                    "flat_sections": self._flat_sections.get(doc_id),
                    "created_at": doc.get("created_at", time.time()),
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────
    # Node-map / flat-sections
    # ──────────────────────────────────────────────────────────────────────
    def get_node_map(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._node_maps.get(doc_id)

    def set_node_map(self, doc_id: str, node_map: Dict[str, Any]) -> None:
        self._node_maps[doc_id] = node_map

    def get_flat_sections(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        return self._flat_sections.get(doc_id)

    def set_flat_sections(self, doc_id: str, sections: List[Dict[str, Any]]) -> None:
        self._flat_sections[doc_id] = sections

    # ──────────────────────────────────────────────────────────────────────
    # Document lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def create_document(
        self,
        doc_id: Optional[str] = None,
        filename: str = "document.pdf",
        status: str = "indexing",
        content_hash: Optional[str] = None,
        file_path: Optional[str] = None,
        current_stage: str = "Processing PDF",
        progress: int = 10,
    ) -> str:
        document_id = doc_id or str(uuid.uuid4())
        now = time.time()
        self._documents[document_id] = {
            "doc_id": document_id,
            "document_id": document_id,
            "filename": filename,
            "status": status,
            "progress": progress,
            "current_stage": current_stage,
            "tree": None,
            "page_count": 0,
            "section_count": 0,
            "indexing_method": None,
            "duration": None,
            "error": None,
            "content_hash": content_hash,
            "file_path": file_path,
            "created_at": now,
            "updated_at": now,
        }
        if content_hash:
            self._hash_to_doc_id[content_hash] = document_id

        # Persist metadata
        persistent_db.upsert_doc_meta(
            doc_id=document_id,
            filename=filename,
            status=status,
            progress=progress,
            current_stage=current_stage,
            content_hash=content_hash,
            file_path=file_path,
        )
        return document_id

    def update_document_progress(
        self,
        doc_id: str,
        status: str = "indexing",
        current_stage: str = "Processing PDF",
        progress: int = 0,
    ) -> None:
        doc = self._documents.get(doc_id)
        if not doc:
            db_meta = persistent_db.get_doc_meta(doc_id)
            if db_meta:
                doc = {
                    "doc_id": doc_id,
                    "document_id": doc_id,
                    "filename": db_meta.get("filename", "document.pdf"),
                    "status": status,
                    "progress": progress,
                    "current_stage": current_stage,
                    "tree": None,
                    "page_count": db_meta.get("page_count", 0),
                    "section_count": db_meta.get("section_count", 0),
                    "indexing_method": db_meta.get("indexing_method"),
                    "duration": db_meta.get("duration"),
                    "error": db_meta.get("error"),
                    "content_hash": db_meta.get("content_hash"),
                    "file_path": db_meta.get("file_path"),
                    "created_at": db_meta.get("created_at", time.time()),
                    "updated_at": time.time(),
                }
                self._documents[doc_id] = doc

        if doc:
            doc["status"] = status
            doc["current_stage"] = current_stage
            doc["progress"] = progress
            doc["updated_at"] = time.time()

            persistent_db.upsert_doc_meta(
                doc_id=doc_id,
                filename=doc.get("filename", "document.pdf"),
                status=status,
                page_count=doc.get("page_count", 0),
                section_count=doc.get("section_count", 0),
                progress=progress,
                current_stage=current_stage,
                indexing_method=doc.get("indexing_method"),
                duration=doc.get("duration"),
                error=doc.get("error"),
                content_hash=doc.get("content_hash"),
                file_path=doc.get("file_path"),
            )

    def set_document_ready(
        self,
        doc_id: str,
        tree: Any,
        page_count: Optional[int] = None,
        section_count: Optional[int] = None,
        indexing_method: Optional[str] = None,
        duration: Optional[float] = None,
        filename: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        if doc_id not in self._documents:
            self.create_document(doc_id=doc_id, filename=filename or "document.pdf", status="ready")

        doc = self._documents[doc_id]
        doc["status"] = "ready"
        doc["progress"] = 100
        doc["current_stage"] = "Ready"
        doc["tree"] = tree
        if page_count is not None:
            doc["page_count"] = page_count
        if section_count is not None:
            doc["section_count"] = section_count
        if indexing_method is not None:
            doc["indexing_method"] = indexing_method
        if duration is not None:
            doc["duration"] = duration
        if filename:
            doc["filename"] = filename
        if file_path:
            doc["file_path"] = file_path
        doc["updated_at"] = time.time()
        doc["error"] = None
        self._trees[doc_id] = tree

        persistent_db.upsert_doc_meta(
            doc_id=doc_id,
            filename=doc["filename"],
            status="ready",
            page_count=doc.get("page_count", 0),
            section_count=doc.get("section_count", 0),
            progress=100,
            current_stage="Ready",
            indexing_method=doc.get("indexing_method"),
            duration=doc.get("duration"),
            error=None,
            content_hash=doc.get("content_hash"),
            file_path=file_path or doc.get("file_path"),
        )

    def set_document_error(
        self,
        doc_id: str,
        error_message: str,
        duration: Optional[float] = None,
    ) -> None:
        if doc_id not in self._documents:
            self.create_document(doc_id=doc_id, status="failed")
        doc = self._documents[doc_id]
        doc["status"] = "failed"
        doc["progress"] = 0
        doc["current_stage"] = "Failed"
        doc["error"] = error_message
        if duration is not None:
            doc["duration"] = duration
        doc["updated_at"] = time.time()

        persistent_db.upsert_doc_meta(
            doc_id=doc_id,
            filename=doc.get("filename", "document.pdf"),
            status="failed",
            page_count=doc.get("page_count", 0),
            section_count=doc.get("section_count", 0),
            progress=0,
            current_stage="Failed",
            indexing_method=doc.get("indexing_method"),
            duration=duration or doc.get("duration"),
            error=error_message,
            content_hash=doc.get("content_hash"),
            file_path=doc.get("file_path"),
        )

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if not doc_id:
            return None
        doc = self._documents.get(doc_id)
        if doc:
            return doc

        # Check SQLite
        db_meta = persistent_db.get_doc_meta(doc_id)
        if db_meta:
            doc_obj = {
                "doc_id": doc_id,
                "document_id": doc_id,
                "filename": db_meta.get("filename", "document.pdf"),
                "status": db_meta.get("status", "unknown"),
                "progress": db_meta.get("progress", 100 if db_meta.get("status") == "ready" else 0),
                "current_stage": db_meta.get("current_stage", "Ready" if db_meta.get("status") == "ready" else "unknown"),
                "page_count": db_meta.get("page_count", 0),
                "section_count": db_meta.get("section_count", 0),
                "indexing_method": db_meta.get("indexing_method"),
                "duration": db_meta.get("duration"),
                "error": db_meta.get("error"),
                "content_hash": db_meta.get("content_hash"),
                "file_path": db_meta.get("file_path"),
                "created_at": db_meta.get("created_at", time.time()),
                "updated_at": db_meta.get("updated_at", time.time()),
            }
            if doc_obj["status"] == "ready":
                cache_path = os.path.join(CACHE_DIR, f"{doc_obj['content_hash']}.json") if doc_obj.get("content_hash") else None
                if cache_path and os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        doc_obj["tree"] = cached.get("tree")
                        self._trees[doc_id] = cached.get("tree")
                        if cached.get("node_map"):
                            self._node_maps[doc_id] = cached["node_map"]
                        if cached.get("flat_sections"):
                            self._flat_sections[doc_id] = cached["flat_sections"]
                    except Exception:
                        pass
            self._documents[doc_id] = doc_obj
            return doc_obj
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Legacy / convenience tree helpers
    # ──────────────────────────────────────────────────────────────────────
    def save_tree(self, tree: Any, doc_id: Optional[str] = None, filename: str = "document.pdf") -> str:
        document_id = doc_id or str(uuid.uuid4())
        self.set_document_ready(document_id, tree=tree, filename=filename)
        return document_id

    def get_tree(self, doc_id: str) -> Optional[Any]:
        doc = self._documents.get(doc_id)
        if doc and doc.get("tree") is not None:
            return doc["tree"]
        return self._trees.get(doc_id)

    def tree_exists(self, doc_id: str) -> bool:
        if doc_id in self._documents and self._documents[doc_id].get("status") == "ready" and self._documents[doc_id].get("tree"):
            return True
        return doc_id in self._trees

    # ──────────────────────────────────────────────────────────────────────
    # Chat operations  (all backed by SQLite now)
    # ──────────────────────────────────────────────────────────────────────
    def create_chat(self, chat_id: Optional[str] = None, title: Optional[str] = None) -> str:
        cid = chat_id or str(uuid.uuid4())
        persistent_db.create_chat(chat_id=cid, title=title)
        return cid

    def get_chat_history(self, chat_id: str) -> Optional[List[Dict[str, str]]]:
        meta = persistent_db.get_chat(chat_id)
        if meta is None:
            return None
        return persistent_db.get_messages(chat_id)

    def chat_exists(self, chat_id: str) -> bool:
        return persistent_db.get_chat(chat_id) is not None

    def ensure_chat(self, chat_id: str) -> List[Dict[str, str]]:
        if not self.chat_exists(chat_id):
            persistent_db.create_chat(chat_id=chat_id)
        return persistent_db.get_messages(chat_id)

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        persistent_db.add_message(chat_id=chat_id, role=role, content=content)

    def update_chat_history(self, chat_id: str, history: List[Dict[str, str]]) -> None:
        persistent_db.clear_messages(chat_id)
        for msg in history:
            persistent_db.add_message(chat_id=chat_id, role=msg["role"], content=msg["content"])

    def set_chat_doc(self, chat_id: str, doc_id: str, doc_name: Optional[str] = None) -> None:
        # ensure the chat row exists first
        if not self.chat_exists(chat_id):
            persistent_db.create_chat(chat_id=chat_id)
        # look up doc name if not supplied
        if doc_name is None:
            doc = self._documents.get(doc_id)
            doc_name = doc.get("filename") if doc else None
        persistent_db.set_chat_doc(chat_id=chat_id, doc_id=doc_id, doc_name=doc_name)

    def get_chat_doc(self, chat_id: str) -> Optional[str]:
        meta = persistent_db.get_chat(chat_id)
        return meta["doc_id"] if meta else None

    def remove_chat_doc(self, chat_id: str) -> None:
        persistent_db.remove_chat_doc(chat_id)

    def delete_chat(self, chat_id: str) -> None:
        persistent_db.delete_chat(chat_id)

    def list_chats(self) -> List[Dict[str, Any]]:
        return persistent_db.list_chats()

    def update_chat_title(self, chat_id: str, title: str) -> None:
        persistent_db.update_chat_title(chat_id=chat_id, title=title)


# Global singleton storage instance
storage = MemoryStorage()
