from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    sqlite_path: Path = PROJECT_ROOT / "data" / "wiki_rag.sqlite3"
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "wikipedia_chunks"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    max_distance: float = 0.65


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "llama3.2:3b"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        sqlite_path=Path(os.getenv("SQLITE_PATH", PROJECT_ROOT / "data" / "wiki_rag.sqlite3")),
        chroma_path=Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / "data" / "chroma")),
        collection_name=os.getenv("CHROMA_COLLECTION", "wikipedia_chunks"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "5")),
        max_distance=float(os.getenv("MAX_DISTANCE", "0.65")),
    )

