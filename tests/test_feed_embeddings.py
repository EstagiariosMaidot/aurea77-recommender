"""Tests for feed embeddings pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.feed_embeddings import embed_posts, build_post_text


def test_build_post_text_uses_body(monkeypatch):
    row = pd.Series({"id": 1, "body": "A trail run in the mountains.", "privacy": "public"})
    text = build_post_text(row)
    assert "trail run" in text.lower()


def test_build_post_text_handles_empty_body():
    row = pd.Series({"id": 42, "body": None, "privacy": "public"})
    text = build_post_text(row)
    # Must return a non-empty string so the embedding is well-defined
    assert isinstance(text, str) and len(text) > 0


def test_embed_posts_returns_expected_shape():
    posts = pd.DataFrame([
        {"id": 1, "body": "Trail running is amazing.", "privacy": "public"},
        {"id": 2, "body": "Marathon race report.", "privacy": "public"},
        {"id": 3, "body": None, "privacy": "public"},
    ])

    matrix, id_to_row = embed_posts(posts)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 384)  # 384 = model dim
    assert id_to_row == {1: 0, 2: 1, 3: 2}


def test_embed_posts_is_deterministic():
    posts = pd.DataFrame([
        {"id": 1, "body": "The same text.", "privacy": "public"},
        {"id": 2, "body": "The same text.", "privacy": "public"},
    ])

    matrix, _ = embed_posts(posts)

    # Same text produces (essentially) same embedding
    np.testing.assert_allclose(matrix[0], matrix[1], rtol=1e-5, atol=1e-6)
