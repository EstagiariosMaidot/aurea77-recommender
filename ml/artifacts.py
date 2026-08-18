"""Model artifact I/O.

Each trained model lives under a single directory containing:
- model.pkl              (joblib-serialized)
- event_embeddings.npy   (n_events, 384) float32
- event_id_to_row.json   {event_id_as_string: row_index}
- metadata.json
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class ArtifactStore:
    model: Any
    event_embeddings: np.ndarray
    event_id_to_row: dict[int, int]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "ArtifactStore":
        root = Path(root)
        model_path = root / "model.pkl"
        emb_path = root / "event_embeddings.npy"
        map_path = root / "event_id_to_row.json"
        meta_path = root / "metadata.json"
        for p in (model_path, emb_path, map_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(p)

        model = joblib.load(model_path)
        emb = np.load(emb_path)
        id_to_row = {int(k): int(v) for k, v in json.loads(map_path.read_text()).items()}
        metadata = json.loads(meta_path.read_text())
        return cls(model=model, event_embeddings=emb,
                   event_id_to_row=id_to_row, metadata=metadata)


class CachedArtifactStore:
    """Thread-safe in-memory cache that reloads when artifacts change on disk."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._lock = threading.Lock()
        self._store: ArtifactStore | None = None
        self._mtime: float = 0.0

    def _meta_mtime(self) -> float:
        """Use metadata.json mtime as change indicator."""
        try:
            return (self._root / "metadata.json").stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def get(self) -> ArtifactStore:
        """Return cached store, reloading only when files changed."""
        current_mtime = self._meta_mtime()
        if self._store is not None and current_mtime == self._mtime:
            return self._store
        with self._lock:
            # Double-check after acquiring lock
            current_mtime = self._meta_mtime()
            if self._store is not None and current_mtime == self._mtime:
                return self._store
            self._store = ArtifactStore.load(self._root)
            self._mtime = current_mtime
            return self._store

    def invalidate(self) -> None:
        """Force reload on next access (call after sync/retrain)."""
        with self._lock:
            self._mtime = 0.0


def save_artifacts(
    root: Path,
    *,
    model: Any,
    event_embeddings: np.ndarray,
    event_ids: list[int],
    metadata: dict[str, Any],
) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, root / "model.pkl")
    np.save(root / "event_embeddings.npy", event_embeddings.astype(np.float32))
    id_to_row = {str(eid): i for i, eid in enumerate(event_ids)}
    (root / "event_id_to_row.json").write_text(json.dumps(id_to_row, indent=2))
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))


def upsert_event_embedding(root: Path, event_id: int, embedding: np.ndarray) -> None:
    """Add or update a single event embedding in the artifacts directory.

    If the event already exists, its embedding is replaced in-place.
    If it's a new event, the embedding is appended.
    """
    root = Path(root)
    emb_path = root / "event_embeddings.npy"
    map_path = root / "event_id_to_row.json"

    embeddings = np.load(emb_path)
    id_to_row: dict[int, int] = {
        int(k): int(v) for k, v in json.loads(map_path.read_text()).items()
    }

    if event_id in id_to_row:
        # Update existing
        row_idx = id_to_row[event_id]
        embeddings[row_idx] = embedding.astype(np.float32)
    else:
        # Append new
        row_idx = embeddings.shape[0]
        embeddings = np.vstack([embeddings, embedding.astype(np.float32).reshape(1, -1)])
        id_to_row[event_id] = row_idx

    np.save(emb_path, embeddings)
    (map_path).write_text(json.dumps({str(k): v for k, v in id_to_row.items()}, indent=2))


def remove_event_embedding(root: Path, event_id: int) -> bool:
    """Remove an event embedding. Returns True if it existed."""
    root = Path(root)
    emb_path = root / "event_embeddings.npy"
    map_path = root / "event_id_to_row.json"

    id_to_row: dict[int, int] = {
        int(k): int(v) for k, v in json.loads(map_path.read_text()).items()
    }

    if event_id not in id_to_row:
        return False

    row_idx = id_to_row.pop(event_id)
    embeddings = np.load(emb_path)
    embeddings = np.delete(embeddings, row_idx, axis=0)

    # Reindex: all rows after the deleted one shift down by 1
    for eid, idx in id_to_row.items():
        if idx > row_idx:
            id_to_row[eid] = idx - 1

    np.save(emb_path, embeddings)
    (map_path).write_text(json.dumps({str(k): v for k, v in id_to_row.items()}, indent=2))
    return True
