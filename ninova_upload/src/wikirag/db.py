from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('person', 'place')),
    source_url TEXT NOT NULL,
    text TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    page_title TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('person', 'place')),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    source_url TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(page_title) REFERENCES pages(title)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def upsert_page(
    connection: sqlite3.Connection,
    title: str,
    entity_type: str,
    source_url: str,
    text: str,
) -> None:
    connection.execute(
        """
        INSERT INTO pages(title, entity_type, source_url, text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            entity_type=excluded.entity_type,
            source_url=excluded.source_url,
            text=excluded.text,
            fetched_at=CURRENT_TIMESTAMP
        """,
        (title, entity_type, source_url, text),
    )
    connection.commit()


def replace_chunks(
    connection: sqlite3.Connection,
    page_title: str,
    entity_type: str,
    source_url: str,
    chunks: Iterable[str],
) -> list[dict[str, str | int]]:
    connection.execute("DELETE FROM chunks WHERE page_title = ?", (page_title,))
    rows: list[dict[str, str | int]] = []
    safe_title = page_title.lower().replace(" ", "_").replace("/", "_")

    for index, chunk in enumerate(chunks):
        chunk_id = f"{entity_type}:{safe_title}:{index}"
        row = {
            "id": chunk_id,
            "page_title": page_title,
            "entity_type": entity_type,
            "chunk_index": index,
            "text": chunk,
            "source_url": source_url,
        }
        rows.append(row)
        connection.execute(
            """
            INSERT INTO chunks(id, page_title, entity_type, chunk_index, text, source_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                page_title,
                entity_type,
                index,
                chunk,
                source_url,
            ),
        )

    connection.commit()
    return rows


def count_pages(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) AS count FROM pages").fetchone()
    return int(row["count"])

