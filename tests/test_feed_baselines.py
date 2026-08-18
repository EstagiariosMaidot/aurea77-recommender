"""Tests for baseline rankers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ml.feed_baselines import (
    rank_chronological,
    rank_popularity,
    rank_friends_only,
)


NOW = datetime(2026, 7, 10, 12, 0, 0)


def _make_posts() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "user_id": 10, "likes_count": 0, "comments_count": 0, "created_at": pd.Timestamp(NOW - timedelta(days=5))},
        {"id": 2, "user_id": 10, "likes_count": 3, "comments_count": 2, "created_at": pd.Timestamp(NOW - timedelta(days=3))},
        {"id": 3, "user_id": 20, "likes_count": 10, "comments_count": 0, "created_at": pd.Timestamp(NOW - timedelta(days=1))},
        {"id": 4, "user_id": 30, "likes_count": 1, "comments_count": 0, "created_at": pd.Timestamp(NOW - timedelta(hours=6))},
    ])


def test_chronological_returns_ids_newest_first():
    posts = _make_posts()
    ranked = rank_chronological(candidates=[1, 2, 3, 4], posts=posts)
    assert ranked == [4, 3, 2, 1]


def test_popularity_returns_ids_by_engagement_desc():
    posts = _make_posts()
    # engagement: 1→0, 2→5, 3→10, 4→1
    ranked = rank_popularity(candidates=[1, 2, 3, 4], posts=posts)
    assert ranked == [3, 2, 4, 1]


def test_friends_only_drops_non_friends_then_orders_chronologically():
    posts = _make_posts()
    friend_ids = {10}  # only user 10 is a friend
    ranked = rank_friends_only(
        candidates=[1, 2, 3, 4],
        posts=posts,
        friend_ids=friend_ids,
    )
    # Only ids 1, 2 (both authored by user 10) survive; newer first.
    assert ranked == [2, 1]


def test_baselines_handle_empty_candidates():
    posts = _make_posts()
    assert rank_chronological(candidates=[], posts=posts) == []
    assert rank_popularity(candidates=[], posts=posts) == []
    assert rank_friends_only(candidates=[], posts=posts, friend_ids={10}) == []
