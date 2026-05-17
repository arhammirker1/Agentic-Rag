"""
local.py — Local filesystem storage backend for PageIndex trees.

This is the default backend.  Trees are saved as individual JSON files
inside a configurable directory.  Zero external dependencies.

Usage:
    store = LocalStore("./data/trees")
    store.save("annual_report_2024", tree_dict)
    tree = store.load("annual_report_2024")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base import TreeStore


class LocalStore(TreeStore):
    """
    Store PageIndex trees as JSON files on the local filesystem.

    Parameters
    ----------
    directory : str or Path
        Root directory where tree JSON files are stored.
        Created automatically if it doesn't exist.
    """

    def __init__(self, directory: str | Path = "./pageindex_data/trees"):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        # Sanitise doc_id for filesystem safety
        safe = doc_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.json"

    def save(self, doc_id: str, tree: Dict[str, Any]) -> None:
        path = self._path(doc_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

    def load(self, doc_id: str) -> Dict[str, Any]:
        path = self._path(doc_id)
        if not path.exists():
            raise FileNotFoundError(f"No tree stored for doc_id='{doc_id}' at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self, doc_id: str) -> None:
        path = self._path(doc_id)
        if path.exists():
            path.unlink()

    def exists(self, doc_id: str) -> bool:
        return self._path(doc_id).exists()

    def list_ids(self) -> List[str]:
        return [p.stem for p in self._dir.glob("*.json")]

    def __repr__(self) -> str:
        return f"LocalStore('{self._dir}')"
