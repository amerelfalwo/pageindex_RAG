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
DATA_DIR = "/data" if (os.path.exists("/data") and os.access("/data", os.W_OK)) else os.path.join(_APP_DIR, "data")
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
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT,
    status          TEXT,
    page_count      INTEGER DEFAULT 0,
    section_count   INTEGER DEFAULT 0,
    progress        INTEGER DEFAULT 0,
    current_stage   TEXT DEFAULT 'uploaded',
    indexing_method TEXT,
    duration        REAL,
    error           TEXT,
    content_hash    TEXT,
    file_path       TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_documents (
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL REFERENCES doc_meta(doc_id) ON DELETE CASCADE,
    doc_name    TEXT,
    attached_at REAL NOT NULL,
    PRIMARY KEY (chat_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_chat_docs ON chat_documents(chat_id);

CREATE TABLE IF NOT EXISTS chat_summaries (
    chat_id     TEXT PRIMARY KEY REFERENCES chats(chat_id) ON DELETE CASCADE,
    summary     TEXT NOT NULL,
    turn_count  INTEGER DEFAULT 0,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_memory (
    doc_id            TEXT PRIMARY KEY REFERENCES doc_meta(doc_id) ON DELETE CASCADE,
    filename          TEXT NOT NULL,
    executive_summary TEXT,
    key_entities      TEXT,
    key_findings      TEXT,
    topics            TEXT,
    updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS global_knowledge_summary (
    id                  TEXT PRIMARY KEY,
    collective_summary  TEXT NOT NULL,
    total_documents     INTEGER DEFAULT 0,
    key_insights        TEXT,
    updated_at          REAL NOT NULL
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
            # Ensure newly added columns exist in older DB instances
            cursor = conn.execute("PRAGMA table_info(doc_meta)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            for col, col_type, default_val in [
                ("section_count", "INTEGER", "0"),
                ("progress", "INTEGER", "0"),
                ("current_stage", "TEXT", "'uploaded'"),
                ("indexing_method", "TEXT", "NULL"),
                ("duration", "REAL", "NULL"),
                ("error", "TEXT", "NULL"),
            ]:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE doc_meta ADD COLUMN {col} {col_type} DEFAULT {default_val}")

            # Migrate existing chats.doc_id into chat_documents table
            conn.execute("""
                INSERT OR IGNORE INTO chat_documents(chat_id, doc_id, doc_name, attached_at)
                SELECT chat_id, doc_id, doc_name, created_at FROM chats
                WHERE doc_id IS NOT NULL AND doc_id != ''
            """)

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
                "INSERT OR IGNORE INTO chats(chat_id, title, created_at, updated_at) VALUES(?, 'New research', ?, ?)",
                (chat_id, now, now),
            )
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
        section_count: int = 0,
        progress: int = 0,
        current_stage: str = "uploaded",
        indexing_method: Optional[str] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        content_hash: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO doc_meta(
                    doc_id, filename, status, page_count, section_count,
                    progress, current_stage, indexing_method, duration,
                    error, content_hash, file_path, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    status=excluded.status,
                    page_count=excluded.page_count,
                    section_count=excluded.section_count,
                    progress=excluded.progress,
                    current_stage=excluded.current_stage,
                    indexing_method=COALESCE(excluded.indexing_method, doc_meta.indexing_method),
                    duration=COALESCE(excluded.duration, doc_meta.duration),
                    error=excluded.error,
                    content_hash=COALESCE(excluded.content_hash, doc_meta.content_hash),
                    file_path=COALESCE(excluded.file_path, doc_meta.file_path),
                    updated_at=excluded.updated_at
                """,
                (
                    doc_id, filename, status, page_count, section_count,
                    progress, current_stage, indexing_method, duration,
                    error, content_hash, file_path, now, now
                ),
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
        # Match latest document with this hash
        row = self._conn().execute(
            "SELECT * FROM doc_meta WHERE content_hash=? ORDER BY updated_at DESC LIMIT 1",
            (content_hash,),
        ).fetchone()
        return dict(row) if row else None

    def delete_doc(self, doc_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM doc_meta WHERE doc_id=?", (doc_id,))
            conn.execute("UPDATE chats SET doc_id=NULL, doc_name=NULL WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM chat_documents WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM knowledge_memory WHERE doc_id=?", (doc_id,))
            return cur.rowcount > 0

    # ── multi-doc session associations ──────────────────────────────────────
    def add_chat_document(self, chat_id: str, doc_id: str, doc_name: Optional[str] = None) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO chat_documents(chat_id, doc_id, doc_name, attached_at)
                VALUES(?,?,?,?)
                ON CONFLICT(chat_id, doc_id) DO UPDATE SET
                    doc_name=COALESCE(excluded.doc_name, chat_documents.doc_name),
                    attached_at=excluded.attached_at
                """,
                (chat_id, doc_id, doc_name, now),
            )
            # Maintain backward compatibility on chats table with latest doc
            conn.execute(
                "UPDATE chats SET doc_id=?, doc_name=?, updated_at=? WHERE chat_id=?",
                (doc_id, doc_name, now, chat_id),
            )

    def remove_chat_document(self, chat_id: str, doc_id: str) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute("DELETE FROM chat_documents WHERE chat_id=? AND doc_id=?", (chat_id, doc_id))
            rem = conn.execute(
                "SELECT doc_id, doc_name FROM chat_documents WHERE chat_id=? ORDER BY attached_at DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
            if rem:
                conn.execute(
                    "UPDATE chats SET doc_id=?, doc_name=?, updated_at=? WHERE chat_id=?",
                    (rem["doc_id"], rem["doc_name"], now, chat_id),
                )
            else:
                conn.execute(
                    "UPDATE chats SET doc_id=NULL, doc_name=NULL, updated_at=? WHERE chat_id=?",
                    (now, chat_id),
                )

    def get_chat_documents(self, chat_id: str) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            """
            SELECT cd.chat_id, cd.doc_id, cd.doc_name, cd.attached_at,
                   dm.status, dm.page_count, dm.section_count, dm.filename
            FROM chat_documents cd
            LEFT JOIN doc_meta dm ON cd.doc_id = dm.doc_id
            WHERE cd.chat_id=?
            ORDER BY cd.attached_at ASC
            """,
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_chat_documents(self, chat_id: str) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute("DELETE FROM chat_documents WHERE chat_id=?", (chat_id,))
            conn.execute("UPDATE chats SET doc_id=NULL, doc_name=NULL, updated_at=? WHERE chat_id=?", (now, chat_id))

    # ── session long memory & rolling summaries ─────────────────────────────
    def set_chat_summary(self, chat_id: str, summary: str, turn_count: int = 0) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO chat_summaries(chat_id, summary, turn_count, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    summary=excluded.summary,
                    turn_count=excluded.turn_count,
                    updated_at=excluded.updated_at
                """,
                (chat_id, summary, turn_count, now),
            )

    def get_chat_summary(self, chat_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM chat_summaries WHERE chat_id=?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── global knowledge memory & document dossiers ────────────────────────
    def save_knowledge_dossier(
        self,
        doc_id: str,
        filename: str,
        executive_summary: str,
        key_entities: Optional[List[str]] = None,
        key_findings: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
    ) -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_memory(
                    doc_id, filename, executive_summary, key_entities, key_findings, topics, updated_at
                )
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    executive_summary=excluded.executive_summary,
                    key_entities=excluded.key_entities,
                    key_findings=excluded.key_findings,
                    topics=excluded.topics,
                    updated_at=excluded.updated_at
                """,
                (
                    doc_id,
                    filename,
                    executive_summary,
                    json.dumps(key_entities or []),
                    json.dumps(key_findings or []),
                    json.dumps(topics or []),
                    now,
                ),
            )

    def get_knowledge_dossier(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM knowledge_memory WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        for k in ("key_entities", "key_findings", "topics"):
            if res.get(k) and isinstance(res[k], str):
                try:
                    res[k] = json.loads(res[k])
                except Exception:
                    res[k] = []
        return res

    def list_knowledge_dossiers(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM knowledge_memory ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            for k in ("key_entities", "key_findings", "topics"):
                if item.get(k) and isinstance(item[k], str):
                    try:
                        item[k] = json.loads(item[k])
                    except Exception:
                        item[k] = []
            result.append(item)
        return result

    def delete_knowledge_dossier(self, doc_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM knowledge_memory WHERE doc_id=?", (doc_id,))

    def save_global_knowledge_summary(
        self,
        collective_summary: str,
        total_documents: int,
        key_insights: Optional[List[str]] = None,
    ) -> None:
        now = time.time()
        gid = "global"
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO global_knowledge_summary(
                    id, collective_summary, total_documents, key_insights, updated_at
                )
                VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    collective_summary=excluded.collective_summary,
                    total_documents=excluded.total_documents,
                    key_insights=excluded.key_insights,
                    updated_at=excluded.updated_at
                """,
                (gid, collective_summary, total_documents, json.dumps(key_insights or []), now),
            )

    def get_global_knowledge_summary(self) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM global_knowledge_summary WHERE id='global'"
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("key_insights") and isinstance(res["key_insights"], str):
            try:
                res["key_insights"] = json.loads(res["key_insights"])
            except Exception:
                res["key_insights"] = []
        return res



# Global singleton
persistent_db = PersistentDB()
