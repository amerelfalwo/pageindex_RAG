import os
import json
import time
import uuid
from typing import Dict, List, Optional, Any

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "index_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class MemoryStorage:
    """In-memory storage for document states, trees, and chat sessions with disk-backed hash caching."""

    def __init__(self) -> None:
        self._trees: Dict[str, Any] = {}
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._chats: Dict[str, List[Dict[str, str]]] = {}
        self._chat_docs: Dict[str, str] = {}
        self._hash_to_doc_id: Dict[str, str] = {}
        self._node_maps: Dict[str, Dict[str, Any]] = {}
        self._flat_sections: Dict[str, List[Dict[str, Any]]] = {}

    # Document hash caching with persistent disk fallback
    def get_doc_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        doc_id = self._hash_to_doc_id.get(content_hash)
        if doc_id and doc_id in self._documents:
            doc = self._documents[doc_id]
            if doc.get("status") == "ready" and doc.get("tree"):
                return doc

        # Check persistent disk cache
        cache_path = os.path.join(CACHE_DIR, f"{content_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                d_id = cached_data["doc_id"]
                self._documents[d_id] = {
                    "doc_id": d_id,
                    "filename": cached_data.get("filename", "document.pdf"),
                    "status": "ready",
                    "tree": cached_data.get("tree"),
                    "page_count": cached_data.get("page_count", 1),
                    "error": None,
                    "content_hash": content_hash,
                    "created_at": cached_data.get("created_at", time.time()),
                    "updated_at": time.time(),
                }
                self._trees[d_id] = cached_data.get("tree")
                if cached_data.get("node_map"):
                    self._node_maps[d_id] = cached_data["node_map"]
                if cached_data.get("flat_sections"):
                    self._flat_sections[d_id] = cached_data["flat_sections"]
                self._hash_to_doc_id[content_hash] = d_id
                return self._documents[d_id]
            except Exception:
                pass

        return None

    def set_hash_mapping(self, content_hash: str, doc_id: str) -> None:
        self._hash_to_doc_id[content_hash] = doc_id
        # Persist to disk cache
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

    # Pre-indexed node lookup maps for fast O(1) retrieval
    def get_node_map(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._node_maps.get(doc_id)

    def set_node_map(self, doc_id: str, node_map: Dict[str, Any]) -> None:
        self._node_maps[doc_id] = node_map

    def get_flat_sections(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        return self._flat_sections.get(doc_id)

    def set_flat_sections(self, doc_id: str, sections: List[Dict[str, Any]]) -> None:
        self._flat_sections[doc_id] = sections

    # Document state lifecycle operations
    def create_document(
        self,
        doc_id: Optional[str] = None,
        filename: str = "document.pdf",
        status: str = "indexing",
        content_hash: Optional[str] = None,
    ) -> str:
        document_id = doc_id or str(uuid.uuid4())
        self._documents[document_id] = {
            "doc_id": document_id,
            "filename": filename,
            "status": status,  # "indexing" | "ready" | "error"
            "tree": None,
            "page_count": 0,
            "error": None,
            "content_hash": content_hash,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        if content_hash:
            self._hash_to_doc_id[content_hash] = document_id
        return document_id

    def set_document_ready(
        self,
        doc_id: str,
        tree: Any,
        page_count: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> None:
        if doc_id not in self._documents:
            self.create_document(doc_id=doc_id, filename=filename or "document.pdf", status="ready")

        doc = self._documents[doc_id]
        doc["status"] = "ready"
        doc["tree"] = tree
        doc["page_count"] = page_count if page_count is not None else (len(tree) if isinstance(tree, list) else 1)
        if filename:
            doc["filename"] = filename
        doc["updated_at"] = time.time()
        doc["error"] = None

        # Keep _trees in sync
        self._trees[doc_id] = tree

    def set_document_error(self, doc_id: str, error_message: str) -> None:
        if doc_id not in self._documents:
            self.create_document(doc_id=doc_id, status="error")
        doc = self._documents[doc_id]
        doc["status"] = "error"
        doc["error"] = error_message
        doc["updated_at"] = time.time()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._documents.get(doc_id)

    # Legacy & convenience tree operations
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

    # Chat operations
    def create_chat(self) -> str:
        chat_id = str(uuid.uuid4())
        self._chats[chat_id] = []
        return chat_id

    def get_chat_history(self, chat_id: str) -> Optional[List[Dict[str, str]]]:
        return self._chats.get(chat_id)

    def chat_exists(self, chat_id: str) -> bool:
        return chat_id in self._chats

    def ensure_chat(self, chat_id: str) -> List[Dict[str, str]]:
        if chat_id not in self._chats:
            self._chats[chat_id] = []
        return self._chats[chat_id]

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        self.ensure_chat(chat_id).append({"role": role, "content": content})

    def update_chat_history(self, chat_id: str, history: List[Dict[str, str]]) -> None:
        self._chats[chat_id] = history

    def set_chat_doc(self, chat_id: str, doc_id: str) -> None:
        self._chat_docs[chat_id] = doc_id

    def get_chat_doc(self, chat_id: str) -> Optional[str]:
        return self._chat_docs.get(chat_id)

    def remove_chat_doc(self, chat_id: str) -> None:
        self._chat_docs.pop(chat_id, None)


# Global singleton storage instance
storage = MemoryStorage()
