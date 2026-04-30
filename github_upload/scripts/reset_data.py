#!/usr/bin/env python
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wikirag.config import load_settings


def main() -> None:
    settings = load_settings()
    if settings.sqlite_path.exists():
        settings.sqlite_path.unlink()
        print(f"Removed {settings.sqlite_path}")
    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)
        print(f"Removed {settings.chroma_path}")
    print("Local data reset complete.")


if __name__ == "__main__":
    main()

