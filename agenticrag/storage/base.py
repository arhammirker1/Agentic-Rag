"""
base.py — Abstract base class for tree storage backends.

All storage backends must implement this interface.
This allows users to swap between local filesystem, GCS, S3,
or any custom backend without changing application code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TreeStore(ABC):
    """
    Abstract interface for storing and retrieving PageIndex tree JSON files.

    Subclasses must implement: save, load, delete, exists, list_ids.

    Each tree is identified by a unique string `doc_id`.
    The tree itself is a JSON-serialisable dict.
    """

    @abstractmethod
    def save(self, doc_id: str, tree: Dict[str, Any]) -> None:
        """
        Persist a tree dict under the given doc_id.

        Parameters
        ----------
        doc_id : Unique identifier for the document.
        tree   : The PageIndex tree dict (JSON-serialisable).
        """
        ...

    @abstractmethod
    def load(self, doc_id: str) -> Dict[str, Any]:
        """
        Load and return the tree dict for doc_id.

        Raises
        ------
        FileNotFoundError if doc_id does not exist.
        """
        ...

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove the stored tree for doc_id (no-op if missing)."""
        ...

    @abstractmethod
    def exists(self, doc_id: str) -> bool:
        """Return True if a tree is stored for doc_id."""
        ...

    @abstractmethod
    def list_ids(self) -> List[str]:
        """Return a list of all stored doc_ids."""
        ...
