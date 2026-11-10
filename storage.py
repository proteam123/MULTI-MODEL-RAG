import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from config import PROJECT_DIR

DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "assistant.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_storage() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                indexed_at TEXT,
                chunk_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                image_paths_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            """
        )


def save_uploaded_pdf(source_name: str, content: bytes) -> dict:
    source_id = hashlib.sha256(content).hexdigest()
    stem = Path(source_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]", "_", stem).strip(" .")
    safe_stem = safe_stem or "document"
    file_path = UPLOAD_DIR / f"{source_id}__{safe_stem}.pdf"
    if not file_path.exists():
        file_path.write_bytes(content)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (source_id, source_name, file_path, uploaded_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                source_name = excluded.source_name,
                file_path = excluded.file_path
            """,
            (source_id, source_name, str(file_path), uploaded_at),
        )
    return {
        "source_id": source_id,
        "source_name": source_name,
        "file_path": str(file_path),
        "uploaded_at": uploaded_at,
    }


def list_recent_documents(limit: int = 20) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT source_id, source_name, file_path, uploaded_at,
                   indexed_at, chunk_count
            FROM documents
            ORDER BY uploaded_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_document_indexed(source_id: str, chunk_count: int) -> None:
    indexed_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE documents
            SET indexed_at = ?, chunk_count = ?
            WHERE source_id = ?
            """,
            (indexed_at, chunk_count, source_id),
        )


def create_chat_session() -> str:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (session_id, title, created_at, updated_at)
            VALUES (?, 'New chat', ?, ?)
            """,
            (session_id, now, now),
        )
    return session_id


def list_chat_sessions() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat_messages(session_id: str) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources_json, image_paths_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY message_id
            """,
            (session_id,),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources_json"]),
            "image_paths": json.loads(row["image_paths_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    image_paths: list[str] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (
                session_id, role, content, sources_json, image_paths_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                json.dumps(sources or []),
                json.dumps(image_paths or []),
                now,
            ),
        )
        title_row = connection.execute(
            "SELECT title FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if title_row is None:
            raise ValueError(f"Chat session not found: {session_id}")
        title = title_row["title"]
        if role == "user" and title == "New chat":
            title = content.strip().replace("\n", " ")[:60] or "New chat"
        connection.execute(
            """
            UPDATE chat_sessions
            SET title = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (title, now, session_id),
        )
