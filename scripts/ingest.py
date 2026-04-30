#!/usr/bin/env python
from __future__ import annotations

import sys
from time import sleep
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wikirag.chunking import chunk_text
from wikirag.config import load_settings
from wikirag.db import connect, replace_chunks, upsert_page
from wikirag.entities import PEOPLE, PLACES
from wikirag.ollama_client import OllamaClient
from wikirag.vector_store import VectorStore
from wikirag.wikipedia import fetch_wikipedia_page


def ingest_entity(
    title: str,
    entity_type: str,
    ollama: OllamaClient,
    vector_store: VectorStore,
) -> int:
    settings = load_settings()
    with connect(settings.sqlite_path) as connection:
        page = fetch_wikipedia_page(title)
        chunks = chunk_text(page.text, settings.chunk_size, settings.chunk_overlap)
        upsert_page(connection, page.title, entity_type, page.source_url, page.text)
        rows = replace_chunks(connection, page.title, entity_type, page.source_url, chunks)
        embeddings = ollama.embed_many([row["text"] for row in rows])
        vector_store.upsert_chunks(rows, embeddings)
        return len(rows)


def main() -> None:
    settings = load_settings()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    ollama = OllamaClient(settings.ollama_base_url, settings.llm_model, settings.embed_model)
    vector_store = VectorStore(settings.chroma_path, settings.collection_name)
    ollama.healthcheck()

    entities = [(title, "person") for title in PEOPLE] + [(title, "place") for title in PLACES]
    total_chunks = 0
    for title, entity_type in entities:
        print(f"Ingesting {entity_type}: {title}")
        try:
            chunk_count = ingest_entity(title, entity_type, ollama, vector_store)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            raise
        total_chunks += chunk_count
        print(f"  stored {chunk_count} chunks")
        sleep(1)

    print(f"Done. Ingested {len(entities)} pages and {total_chunks} chunks.")


if __name__ == "__main__":
    main()
