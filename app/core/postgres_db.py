"""
postgres_db.py
──────────────
PostgreSQL persistence layer for enterprise-grade durability.
Stores chat sessions, messages, document metadata, and heavy document index trees (JSONB).
Even if the container or app restarts or sleeps, all documents, chats, and indices remain intact.
"""

import os
import json
import time
import uuid
import base64
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import psycopg2
import psycopg2.extras
import psycopg2.pool
from utils.logger import app_logger

# Register JSON/JSONB handlers so JSON columns are transparently converted to/from dict/list
psycopg2.extras.register_default_jsonb()

_PG_DDL = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id     TEXT PRIMARY KEY,
    title       TEXT,
    doc_id      TEXT,
    doc_name    TEXT,
    owner_id    TEXT,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pg_chats_owner ON chats(owner_id);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pg_messages_chat ON messages(chat_id, created_at);

CREATE TABLE IF NOT EXISTS doc_meta (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT,
    status          TEXT,
    page_count      INTEGER DEFAULT 0,
    section_count   INTEGER DEFAULT 0,
    progress        INTEGER DEFAULT 0,
    current_stage   TEXT DEFAULT 'uploaded',
    indexing_method TEXT,
    duration        DOUBLE PRECISION,
    error           TEXT,
    content_hash    TEXT,
    file_path       TEXT,
    owner_id        TEXT,
    created_at      DOUBLE PRECISION NOT NULL,
    updated_at      DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pg_doc_meta_hash ON doc_meta(content_hash);
CREATE INDEX IF NOT EXISTS idx_pg_doc_meta_owner ON doc_meta(owner_id);

CREATE TABLE IF NOT EXISTS doc_indices (
    doc_id          TEXT PRIMARY KEY REFERENCES doc_meta(doc_id) ON DELETE CASCADE,
    content_hash    TEXT,
    tree            JSONB,
    node_map        JSONB,
    flat_sections   JSONB,
    file_base64     TEXT,
    created_at      DOUBLE PRECISION NOT NULL,
    updated_at      DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pg_doc_indices_hash ON doc_indices(content_hash);

CREATE TABLE IF NOT EXISTS chat_documents (
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL REFERENCES doc_meta(doc_id) ON DELETE CASCADE,
    doc_name    TEXT,
    attached_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (chat_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_pg_chat_docs ON chat_documents(chat_id);

CREATE TABLE IF NOT EXISTS chat_summaries (
    chat_id     TEXT PRIMARY KEY REFERENCES chats(chat_id) ON DELETE CASCADE,
    summary     TEXT NOT NULL,
    turn_count  INTEGER DEFAULT 0,
    updated_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_memory (
    doc_id            TEXT PRIMARY KEY REFERENCES doc_meta(doc_id) ON DELETE CASCADE,
    filename          TEXT NOT NULL,
    executive_summary TEXT,
    key_entities      JSONB,
    key_findings      JSONB,
    topics            JSONB,
    updated_at        DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS global_knowledge_summary (
    id                  TEXT PRIMARY KEY,
    owner_id            TEXT,
    collective_summary  TEXT NOT NULL,
    total_documents     INTEGER DEFAULT 0,
    key_insights        JSONB,
    updated_at          DOUBLE PRECISION NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pg_gks_owner ON global_knowledge_summary(owner_id);
"""


class PostgresDB:
    """Robust, thread-safe PostgreSQL database client with connection pool and auto-reconnect."""

    def __init__(self, dsn: str):
        # Normalize postgres:// to postgresql:// if needed
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://"):]
        self._dsn = dsn
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._init_pool()
        self._init_schema()

    def _init_pool(self) -> None:
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=self._dsn,
            )
            app_logger.info("[PostgresDB] Connection pool created successfully.")
        except Exception as e:
            app_logger.error(f"[PostgresDB] Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def _conn(self):
        """Context manager that borrows a healthy connection from the pool and handles commits/rollbacks."""
        conn = None
        try:
            conn = self._pool.getconn()
            # Test connection liveness
            if conn.closed:
                conn = psycopg2.connect(self._dsn)
            else:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                except Exception:
                    # Connection is dead, recreate
                    try:
                        self._pool.putconn(conn, close=True)
                    except Exception:
                        pass
                    conn = psycopg2.connect(self._dsn)

            yield conn
            conn.commit()
        except Exception:
            if conn and not conn.closed:
                conn.rollback()
            raise
        finally:
            if conn and self._pool:
                try:
                    self._pool.putconn(conn)
                except Exception:
                    pass

    def _init_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_PG_DDL)
                for tbl, idx in [
                    ("chats", "idx_pg_chats_owner"),
                    ("doc_meta", "idx_pg_doc_meta_owner"),
                    ("global_knowledge_summary", "idx_pg_gks_owner"),
                ]:
                    cur.execute(
                        """
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = 'owner_id';
                        """,
                        (tbl,),
                    )
                    if not cur.fetchone():
                        cur.execute(f"ALTER TABLE {tbl} ADD COLUMN owner_id TEXT;")
                    if tbl == "global_knowledge_summary":
                        cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {idx} ON {tbl}(owner_id);")
                    else:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl}(owner_id);")
        app_logger.info("[PostgresDB] Schema initialized / verified successfully.")

    # ── chat CRUD ───────────────────────────────────────────────────────────
    def create_chat(self, chat_id: Optional[str] = None, title: Optional[str] = None, owner_id: Optional[str] = None) -> str:
        cid = chat_id or str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chats(chat_id, title, owner_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chat_id) DO NOTHING
                    """,
                    (cid, title, owner_id, now, now),
                )
        return cid

    def get_chat(self, chat_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute("SELECT * FROM chats WHERE chat_id = %s AND owner_id = %s", (chat_id, owner_id))
                else:
                    cur.execute("SELECT * FROM chats WHERE chat_id = %s", (chat_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def list_chats(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute("SELECT * FROM chats WHERE owner_id = %s ORDER BY updated_at DESC", (owner_id,))
                else:
                    cur.execute("SELECT * FROM chats ORDER BY updated_at DESC")
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def update_chat_title(self, chat_id: str, title: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chats SET title = %s, updated_at = %s WHERE chat_id = %s",
                    (title, time.time(), chat_id),
                )

    def set_chat_doc(self, chat_id: str, doc_id: str, doc_name: Optional[str] = None) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chats SET doc_id = %s, doc_name = %s, updated_at = %s WHERE chat_id = %s",
                    (doc_id, doc_name, time.time(), chat_id),
                )

    def remove_chat_doc(self, chat_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chats SET doc_id = NULL, doc_name = NULL, updated_at = %s WHERE chat_id = %s",
                    (time.time(), chat_id),
                )

    def delete_chat(self, chat_id: str, owner_id: Optional[str] = None) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if owner_id:
                    cur.execute("DELETE FROM chats WHERE chat_id = %s AND owner_id = %s", (chat_id, owner_id))
                else:
                    cur.execute("DELETE FROM chats WHERE chat_id = %s", (chat_id,))
                return cur.rowcount > 0

    # ── message CRUD ────────────────────────────────────────────────────────
    def add_message(self, chat_id: str, role: str, content: str, owner_id: Optional[str] = None) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chats(chat_id, title, owner_id, created_at, updated_at)
                    VALUES (%s, 'New research', %s, %s, %s)
                    ON CONFLICT (chat_id) DO NOTHING
                    """,
                    (chat_id, owner_id, now, now),
                )
                cur.execute(
                    "INSERT INTO messages(chat_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                    (chat_id, role, content, now),
                )
                cur.execute(
                    "UPDATE chats SET updated_at = %s WHERE chat_id = %s",
                    (now, chat_id),
                )

    def get_messages(self, chat_id: str, limit: int = 0) -> List[Dict[str, str]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if limit > 0:
                    cur.execute(
                        "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
                        (chat_id, limit),
                    )
                    rows = cur.fetchall()
                    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
                cur.execute(
                    "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at ASC",
                    (chat_id,),
                )
                rows = cur.fetchall()
                return [{"role": r["role"], "content": r["content"]} for r in rows]

    def clear_messages(self, chat_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))

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
        owner_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO doc_meta(
                        doc_id, filename, status, page_count, section_count,
                        progress, current_stage, indexing_method, duration,
                        error, content_hash, file_path, owner_id, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        status = EXCLUDED.status,
                        page_count = EXCLUDED.page_count,
                        section_count = EXCLUDED.section_count,
                        progress = EXCLUDED.progress,
                        current_stage = EXCLUDED.current_stage,
                        indexing_method = COALESCE(EXCLUDED.indexing_method, doc_meta.indexing_method),
                        duration = COALESCE(EXCLUDED.duration, doc_meta.duration),
                        error = EXCLUDED.error,
                        content_hash = COALESCE(EXCLUDED.content_hash, doc_meta.content_hash),
                        file_path = COALESCE(EXCLUDED.file_path, doc_meta.file_path),
                        owner_id = COALESCE(EXCLUDED.owner_id, doc_meta.owner_id),
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        doc_id,
                        filename,
                        status,
                        page_count,
                        section_count,
                        progress,
                        current_stage,
                        indexing_method,
                        duration,
                        error,
                        content_hash,
                        file_path,
                        owner_id,
                        now,
                        now,
                    ),
                )

    def get_doc_meta(self, doc_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute("SELECT * FROM doc_meta WHERE doc_id = %s AND owner_id = %s", (doc_id, owner_id))
                else:
                    cur.execute("SELECT * FROM doc_meta WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def list_docs(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute("SELECT * FROM doc_meta WHERE owner_id = %s ORDER BY created_at DESC", (owner_id,))
                else:
                    cur.execute("SELECT * FROM doc_meta ORDER BY created_at DESC")
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def find_doc_by_hash(self, content_hash: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute(
                        "SELECT * FROM doc_meta WHERE content_hash = %s AND owner_id = %s ORDER BY updated_at DESC LIMIT 1",
                        (content_hash, owner_id),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM doc_meta WHERE content_hash = %s ORDER BY updated_at DESC LIMIT 1",
                        (content_hash,),
                    )
                row = cur.fetchone()
                return dict(row) if row else None

    def delete_doc(self, doc_id: str, owner_id: Optional[str] = None) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if owner_id:
                    cur.execute("SELECT doc_id FROM doc_meta WHERE doc_id = %s AND owner_id = %s", (doc_id, owner_id))
                    if not cur.fetchone():
                        return False
                cur.execute("DELETE FROM doc_meta WHERE doc_id = %s", (doc_id,))
                rc = cur.rowcount
                cur.execute("UPDATE chats SET doc_id = NULL, doc_name = NULL WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM chat_documents WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM knowledge_memory WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM doc_indices WHERE doc_id = %s", (doc_id,))
                return rc > 0

    # ── document heavy indices store (JSONB) ─────────────────────────────────
    def save_doc_index(
        self,
        doc_id: str,
        content_hash: Optional[str],
        tree: Any,
        node_map: Optional[Dict[str, Any]] = None,
        flat_sections: Optional[List[Dict[str, Any]]] = None,
        file_base64: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO doc_indices (
                        doc_id, content_hash, tree, node_map, flat_sections, file_base64, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        content_hash = COALESCE(EXCLUDED.content_hash, doc_indices.content_hash),
                        tree = EXCLUDED.tree,
                        node_map = EXCLUDED.node_map,
                        flat_sections = EXCLUDED.flat_sections,
                        file_base64 = COALESCE(EXCLUDED.file_base64, doc_indices.file_base64),
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        doc_id,
                        content_hash,
                        psycopg2.extras.Json(tree) if tree is not None else None,
                        psycopg2.extras.Json(node_map) if node_map is not None else None,
                        psycopg2.extras.Json(flat_sections) if flat_sections is not None else None,
                        file_base64,
                        now,
                        now,
                    ),
                )

    def get_doc_index(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM doc_indices WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_doc_index_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM doc_indices WHERE content_hash = %s ORDER BY updated_at DESC LIMIT 1",
                    (content_hash,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def delete_doc_index(self, doc_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM doc_indices WHERE doc_id = %s", (doc_id,))

    # ── multi-doc session associations ──────────────────────────────────────
    def add_chat_document(self, chat_id: str, doc_id: str, doc_name: Optional[str] = None) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_documents(chat_id, doc_id, doc_name, attached_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chat_id, doc_id) DO UPDATE SET
                        doc_name = COALESCE(EXCLUDED.doc_name, chat_documents.doc_name),
                        attached_at = EXCLUDED.attached_at
                    """,
                    (chat_id, doc_id, doc_name, now),
                )
                # Maintain backward compatibility on chats table
                cur.execute(
                    "UPDATE chats SET doc_id = %s, doc_name = %s, updated_at = %s WHERE chat_id = %s",
                    (doc_id, doc_name, now, chat_id),
                )

    def remove_chat_document(self, chat_id: str, doc_id: str) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("DELETE FROM chat_documents WHERE chat_id = %s AND doc_id = %s", (chat_id, doc_id))
                cur.execute(
                    "SELECT doc_id, doc_name FROM chat_documents WHERE chat_id = %s ORDER BY attached_at DESC LIMIT 1",
                    (chat_id,),
                )
                rem = cur.fetchone()
                if rem:
                    cur.execute(
                        "UPDATE chats SET doc_id = %s, doc_name = %s, updated_at = %s WHERE chat_id = %s",
                        (rem["doc_id"], rem["doc_name"], now, chat_id),
                    )
                else:
                    cur.execute(
                        "UPDATE chats SET doc_id = NULL, doc_name = NULL, updated_at = %s WHERE chat_id = %s",
                        (now, chat_id),
                    )

    def get_chat_documents(self, chat_id: str, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute("SELECT chat_id FROM chats WHERE chat_id = %s AND owner_id = %s", (chat_id, owner_id))
                    if not cur.fetchone():
                        return []
                cur.execute(
                    """
                    SELECT cd.chat_id, cd.doc_id, cd.doc_name, cd.attached_at,
                           dm.status, dm.page_count, dm.section_count, dm.filename
                    FROM chat_documents cd
                    LEFT JOIN doc_meta dm ON cd.doc_id = dm.doc_id
                    WHERE cd.chat_id = %s
                    ORDER BY cd.attached_at ASC
                    """,
                    (chat_id,),
                )
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def clear_chat_documents(self, chat_id: str) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_documents WHERE chat_id = %s", (chat_id,))
                cur.execute(
                    "UPDATE chats SET doc_id = NULL, doc_name = NULL, updated_at = %s WHERE chat_id = %s",
                    (now, chat_id),
                )

    # ── session long memory & rolling summaries ─────────────────────────────
    def set_chat_summary(self, chat_id: str, summary: str, turn_count: int = 0) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_summaries(chat_id, summary, turn_count, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chat_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        turn_count = EXCLUDED.turn_count,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (chat_id, summary, turn_count, now),
                )

    def get_chat_summary(self, chat_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM chat_summaries WHERE chat_id = %s", (chat_id,))
                row = cur.fetchone()
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
        owner_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_memory(
                        doc_id, filename, executive_summary, key_entities, key_findings, topics, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        executive_summary = EXCLUDED.executive_summary,
                        key_entities = EXCLUDED.key_entities,
                        key_findings = EXCLUDED.key_findings,
                        topics = EXCLUDED.topics,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        doc_id,
                        filename,
                        executive_summary,
                        psycopg2.extras.Json(key_entities or []),
                        psycopg2.extras.Json(key_findings or []),
                        psycopg2.extras.Json(topics or []),
                        now,
                    ),
                )

    def get_knowledge_dossier(self, doc_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute(
                        """
                        SELECT km.* FROM knowledge_memory km
                        JOIN doc_meta dm ON km.doc_id = dm.doc_id
                        WHERE km.doc_id = %s AND dm.owner_id = %s
                        """,
                        (doc_id, owner_id),
                    )
                else:
                    cur.execute("SELECT * FROM knowledge_memory WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                if not row:
                    return None
                res = dict(row)
                for k in ("key_entities", "key_findings", "topics"):
                    if res.get(k) is None:
                        res[k] = []
                return res

    def list_knowledge_dossiers(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute(
                        """
                        SELECT km.* FROM knowledge_memory km
                        JOIN doc_meta dm ON km.doc_id = dm.doc_id
                        WHERE dm.owner_id = %s
                        ORDER BY km.updated_at DESC
                        """,
                        (owner_id,),
                    )
                else:
                    cur.execute("SELECT * FROM knowledge_memory ORDER BY updated_at DESC")
                rows = cur.fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    for k in ("key_entities", "key_findings", "topics"):
                        if item.get(k) is None:
                            item[k] = []
                    result.append(item)
                return result

    def delete_knowledge_dossier(self, doc_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM knowledge_memory WHERE doc_id = %s", (doc_id,))

    def save_global_knowledge_summary(
        self,
        collective_summary: str,
        total_documents: int,
        key_insights: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        gid = owner_id or "global"
        with self._conn() as conn:
            with conn.cursor() as cur:
                if owner_id:
                    cur.execute(
                        """
                        INSERT INTO global_knowledge_summary(
                            id, owner_id, collective_summary, total_documents, key_insights, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (owner_id) DO UPDATE SET
                            collective_summary = EXCLUDED.collective_summary,
                            total_documents = EXCLUDED.total_documents,
                            key_insights = EXCLUDED.key_insights,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            gid,
                            owner_id,
                            collective_summary,
                            total_documents,
                            psycopg2.extras.Json(key_insights or []),
                            now,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO global_knowledge_summary(
                            id, collective_summary, total_documents, key_insights, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            collective_summary = EXCLUDED.collective_summary,
                            total_documents = EXCLUDED.total_documents,
                            key_insights = EXCLUDED.key_insights,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            gid,
                            collective_summary,
                            total_documents,
                            psycopg2.extras.Json(key_insights or []),
                            now,
                        ),
                    )

    def get_global_knowledge_summary(self, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if owner_id:
                    cur.execute("SELECT * FROM global_knowledge_summary WHERE owner_id = %s", (owner_id,))
                else:
                    cur.execute("SELECT * FROM global_knowledge_summary WHERE id = 'global'")
                row = cur.fetchone()
                if not row:
                    return None
                res = dict(row)
                if res.get("key_insights") is None:
                    res["key_insights"] = []
                return res
