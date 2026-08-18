"""Sentence-transformer wrapper.

Loads the multilingual model lazily (expensive) and exposes two functions:
- build_event_text: compose the string that gets embedded per event.
- embed_texts:      forward pass → float32 matrix (n_texts, 384).
"""

from __future__ import annotations

import functools
from typing import Iterable

import numpy as np
import pandas as pd


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


@functools.lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_event_text(
    events: pd.DataFrame,
    event_categories: dict[int, set[int]] | None = None,
    slug_by_category_id: dict[int, str] | None = None,
) -> list[str]:
    """Compose event text without relying on the legacy category tables.

    The optional category parameters are retained temporarily for callers on
    the previous function signature, but are intentionally ignored.  Sports
    preference comes from each user's ``athlete_profiles.sports_practiced``.
    """
    texts: list[str] = []
    for row in events.itertuples(index=False):
        desc = (row.description or "").strip()
        parts = [row.name, desc]
        texts.append(". ".join(p for p in parts if p))
    return texts


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Return a (n, 384) float32 matrix of L2-normalised embeddings."""
    model = _get_model()
    emb = model.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


@functools.lru_cache(maxsize=4096)
def embed_sports_profile(sports: tuple[str, ...]) -> np.ndarray:
    """Embed a user's declared sports once and reuse the normalised vector."""
    if not sports:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    text = ". ".join(s.replace("_", " ").replace("-", " ") for s in sports)
    return embed_texts([text])[0]
