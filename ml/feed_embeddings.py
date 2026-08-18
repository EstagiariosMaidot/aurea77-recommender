"""Embed feed posts as dense vectors using the shared sentence-transformer model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.embeddings import embed_texts


def build_post_text(row: pd.Series) -> str:
    """Convert a feed_posts row into the text fed to the embedding model.

    Empty bodies fall back to a neutral placeholder so the embedding remains
    well-defined even when a post is media-only.
    """
    body = row.get("body")
    if body is None or not str(body).strip():
        return "[empty post]"
    return str(body).strip()


def embed_posts(posts: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    """Vectorize every feed post.

    Args:
        posts: DataFrame with at least the columns `id` and `body`.

    Returns:
        matrix: array of shape (len(posts), 384) with one embedding per row.
        id_to_row: mapping from feed_post_id → row index in ``matrix``.
    """
    texts = posts.apply(build_post_text, axis=1).tolist()
    matrix = embed_texts(texts)
    id_to_row = {int(pid): idx for idx, pid in enumerate(posts["id"].tolist())}
    return matrix, id_to_row
