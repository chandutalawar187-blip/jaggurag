"""Persistent OCR-RAG metadata storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


def _connect() -> sqlite3.Connection:
    path = Path(settings.OCR_RAG_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_storage() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_rag_documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                status TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                progress INTEGER DEFAULT 0,
                progress_phase TEXT DEFAULT 'queued',
                error_message TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        for statement in (
            "ALTER TABLE ocr_rag_documents ADD COLUMN progress INTEGER DEFAULT 0",
            "ALTER TABLE ocr_rag_documents ADD COLUMN progress_phase TEXT DEFAULT 'queued'",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_rag_pages (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                confidence REAL,
                image_reference TEXT NOT NULL,
                bounding_boxes_json TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY(document_id) REFERENCES ocr_rag_documents(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


def upsert_document(document: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ocr_rag_documents
            (id, filename, original_path, status, page_count, chunk_count,
             error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(
                (SELECT created_at FROM ocr_rag_documents WHERE id = ?), ?
            ), ?)
            """,
            (
                document["id"], document["filename"], document["original_path"],
                document.get("status", "processing"), document.get("page_count", 0),
                document.get("chunk_count", 0), document.get("error_message", ""),
                document["id"], now, now,
            ),
        )
        conn.commit()


def update_document(document_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE ocr_rag_documents SET {assignments} WHERE id = ?",
            (*fields.values(), document_id),
        )
        conn.commit()


def get_document(document_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ocr_rag_documents WHERE id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row else None


def save_pages(document_id: str, pages: list[dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM ocr_rag_pages WHERE document_id = ?", (document_id,))
        conn.executemany(
            """
            INSERT INTO ocr_rag_pages
            (id, document_id, page_number, text, raw_text, confidence,
             image_reference, bounding_boxes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    page["id"], document_id, page["page_number"], page["text"],
                    page["raw_text"], page.get("confidence"),
                    page["image_reference"],
                    json.dumps(page.get("bounding_boxes", [])),
                )
                for page in pages
            ],
        )
        conn.commit()
