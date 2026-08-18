"""Shared helpers for the feed ML pipeline.

Small utilities that used to be duplicated across ``train_feed``,
``evaluate_feed`` and ``feed_recommender``. Kept intentionally lean —
this module has no side effects and only depends on the standard
scientific stack plus the recommender's own database helper.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from database import open_read_connection


def open_connection(db_path: str | None) -> Any:
    """Open a SQLite file connection when ``db_path`` is given, otherwise
    fall back to the shared MySQL connection opened by
    ``database.open_read_connection``.

    The SQLite path is used exclusively by tests (which pass a tiny fixture DB);
    production reaches the MySQL branch via ``.env``.
    """
    if db_path:
        return sqlite3.connect(db_path)
    return open_read_connection()


def viewer_profile_embedding(
    matrix: np.ndarray,
    id_to_row: dict[int, int],
    viewer_posts: list[int],
) -> np.ndarray:
    """Average the embeddings of the posts the viewer has interacted with.

    Returns a zero vector when the viewer has no known interactions — the
    cold-start fallback used by both training and inference.
    """
    rows = [id_to_row[pid] for pid in viewer_posts if pid in id_to_row]
    if not rows:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return matrix[rows].mean(axis=0)
