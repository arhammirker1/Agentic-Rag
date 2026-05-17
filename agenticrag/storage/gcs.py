"""
gcs.py — Google Cloud Storage backend for PageIndex trees.

Requires:  pip install google-cloud-storage
       or: pip install pageindex[gcs]

Usage:
    store = GCSStore(bucket_name="my-pageindex-bucket", prefix="trees/")
    store.save("annual_report_2024", tree_dict)
    tree = store.load("annual_report_2024")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import TreeStore

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except ImportError:
    raise ImportError(
        "Google Cloud Storage support requires the `google-cloud-storage` package.\n"
        "Install it with:  pip install google-cloud-storage\n"
        "Or:               pip install pageindex[gcs]"
    )


class GCSStore(TreeStore):
    """
    Store PageIndex trees as JSON blobs in Google Cloud Storage.

    Parameters
    ----------
    bucket_name : Name of the GCS bucket.
    prefix      : Optional prefix (folder path) inside the bucket.
    credentials : Optional path to a service account JSON key file.
                  If None, uses Application Default Credentials (ADC).
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "pageindex_trees/",
        credentials: str | None = None,
    ):
        if credentials:
            self._client = gcs_storage.Client.from_service_account_json(credentials)
        else:
            self._client = gcs_storage.Client()

        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix.rstrip("/") + "/"

    def _blob_name(self, doc_id: str) -> str:
        safe = doc_id.replace("/", "_").replace("\\", "_")
        return f"{self._prefix}{safe}.json"

    def save(self, doc_id: str, tree: Dict[str, Any]) -> None:
        blob = self._bucket.blob(self._blob_name(doc_id))
        data = json.dumps(tree, indent=2, ensure_ascii=False)
        blob.upload_from_string(data, content_type="application/json")

    def load(self, doc_id: str) -> Dict[str, Any]:
        blob = self._bucket.blob(self._blob_name(doc_id))
        if not blob.exists():
            raise FileNotFoundError(
                f"No tree stored for doc_id='{doc_id}' in "
                f"gs://{self._bucket.name}/{self._blob_name(doc_id)}"
            )
        data = blob.download_as_text(encoding="utf-8")
        return json.loads(data)

    def delete(self, doc_id: str) -> None:
        blob = self._bucket.blob(self._blob_name(doc_id))
        if blob.exists():
            blob.delete()

    def exists(self, doc_id: str) -> bool:
        return self._bucket.blob(self._blob_name(doc_id)).exists()

    def list_ids(self) -> List[str]:
        blobs = self._client.list_blobs(self._bucket, prefix=self._prefix)
        ids = []
        for blob in blobs:
            name = blob.name.removeprefix(self._prefix)
            if name.endswith(".json"):
                ids.append(name[:-5])  # strip .json
        return ids

    def __repr__(self) -> str:
        return f"GCSStore('{self._bucket.name}/{self._prefix}')"
