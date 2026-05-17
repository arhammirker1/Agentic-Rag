"""
agenticrag.ingestion — Document ingestion and metadata extraction.

Handles the pipeline from raw document → indexed tree + graph metadata.
"""

from .metadata import extract_metadata
from .pipeline import IngestResult, ingest_document

__all__ = ["extract_metadata", "ingest_document", "IngestResult"]
