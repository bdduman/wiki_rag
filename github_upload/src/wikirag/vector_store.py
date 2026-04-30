from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, rows: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not rows:
            return
        if len(rows) != len(embeddings):
            raise ValueError("rows and embeddings must have the same length")

        self.collection.upsert(
            ids=[str(row["id"]) for row in rows],
            documents=[str(row["text"]) for row in rows],
            embeddings=embeddings,
            metadatas=[
                {
                    "title": str(row["page_title"]),
                    "type": str(row["entity_type"]),
                    "source_url": str(row["source_url"]),
                    "chunk_index": int(row["chunk_index"]),
                }
                for row in rows
            ],
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        entity_type: str | None = None,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = []
        if entity_type in {"person", "place"}:
            filters.append({"type": entity_type})
        if title:
            filters.append({"title": title})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}
        else:
            where = None

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        matches = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            matches.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": float(distance),
                }
            )
        return matches

    def get_by_title(
        self,
        title: str,
        entity_type: str | None = None,
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        filters = [{"title": title}]
        if entity_type in {"person", "place"}:
            filters.append({"type": entity_type})
        where = filters[0] if len(filters) == 1 else {"$and": filters}

        result = self.collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        rows = []
        for chunk_id, document, metadata in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            rows.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": 0.0,
                }
            )
        return sorted(
            rows,
            key=lambda row: int(row.get("metadata", {}).get("chunk_index", 999)),
        )[:limit]

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
