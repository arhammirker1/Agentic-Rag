"""
agenticrag.storage — Pluggable storage backends for tree JSON files.

Available backends:
  - LocalStore   : Local filesystem (default, zero dependencies)
  - GCSStore     : Google Cloud Storage (requires `pip install agenticrag[gcs]`)

Usage:
    from agenticrag.storage import LocalStore
    store = LocalStore("./my_trees")
    store.save("doc_id", tree_dict)
    tree = store.load("doc_id")
"""

from .base import TreeStore
from .local import LocalStore

__all__ = ["TreeStore", "LocalStore"]

# Optional backends — import only when available
try:
    from .gcs import GCSStore
    __all__.append("GCSStore")
except ImportError:
    pass
