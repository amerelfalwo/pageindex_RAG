"""
persistent_db.py
────────────────
SQLite-backed persistence layer for long-term memory.
Stores chat sessions, messages, and chat→document associations.
The heavy document trees are kept on-disk in CACHE_DIR (JSON) –
this layer only records the *metadata* needed to reload them.
"""

import os
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# ── paths ──────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_APP_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
os.makedirs(DATA_DIR, exist_ok=True)

# ── schema ─────────────────────────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS chats (
    chat_id     TEXT PRIMARY KEY,
    title       TEXT,
    doc_id      TEXT,
    doc_name    TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);

CREATE TABLE IF NOT EXISTS doc_meta (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT,
    status      TEXT,
    page_count  INTEGER DEFAULT 0,
    content_hash TEXT,
    file_path   TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""


class PersistentDB:
    """Thread-safe SQLite wrapper for long-term chat & document memory."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._path = db_path
        self._local = threading.local()
        self._init_schema()

    # ── connection management ───────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self._tx() as conn:
            conn.executescript(_DDL)

    # ── chat CRUD ───────────────────────────────────────────────────────────
    def create_chat(self, chat_id: Optional[str] = None, title: Optional[str] = None) -> str:
        cid = chat_id or str(uuid.uuid4())
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chats(chat_id, title, created_at, updated_at) VALUES(?,?,?,?)",
                (cid, title, now, now),
            )
        return cid

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM chats WHERE chat_id=?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_chats(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM chats ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_chat_title(self, chat_id: str, title: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE chats SET title=?, updated_at=? WHERE chat_id=?",
                (title, time.time(), chat_id),
            )

    def set_chat_doc(self, chat_id: str, doc_id: str, doc_name: Optional[str] = None) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE chats SET doc_id=?, doc_name=?, updated_at=? WHERE chat_id=?",
                (doc_id, doc_name, time.time(), chat_id),
            )

    def remove_chat_doc(self, chat_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE chats SET doc_id=NULL, doc_name=NULL, updated_at=? WHERE chat_id=?",
                (time.time(), chat_id),
            )

    def delete_chat(self, chat_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))

    # ── message CRUD ────────────────────────────────────────────────────────
    def add_message(self, chat_id: str, role: str, content: str) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES(?,?,?,?)",
                (chat_id, role, content, now),
            )
            conn.execute(
                "UPDATE chats SET updated_at=? WHERE chat_id=?", (now, chat_id)
            )

    def get_messages(self, chat_id: str, limit: int = 0) -> List[Dict[str, str]]:
        q = "SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC"
        if limit > 0:
            rows = self._conn().execute(
                f"SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        rows = self._conn().execute(q, (chat_id,)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def clear_messages(self, chat_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))

    # ── document metadata ────────────────────────────────────────────────────
    def upsert_doc_meta(
        self,
        doc_id: str,
        filename: str,
        status: str,
        page_count: int = 0,
        content_hash: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO doc_meta(doc_id, filename, status, page_count, content_hash, file_path, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    status=excluded.status,
                    page_count=excluded.page_count,
                    content_hash=COALESCE(excluded.content_hash, content_hash),
                    file_path=COALESCE(excluded.file_path, file_path),
                    updated_at=excluded.updated_at
                """,
                (doc_id, filename, status, page_count, content_hash, file_path, now, now),
            )

    def get_doc_meta(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM doc_meta WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_docs(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM doc_meta ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def find_doc_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM doc_meta WHERE content_hash=? AND status='ready'",
            (content_hash,),
        ).fetchone()
        return dict(row) if row else None


# Global singleton
persistent_db = PersistentDB()
