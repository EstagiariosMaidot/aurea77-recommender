"""Tests for feed data loaders (entity queries)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ml.feed_data_loader import (
    load_feed_posts,
    load_users_min,
    load_friendships,
    load_post_media,
    load_user_blocks,
    load_user_mutes,
)
from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def tiny_conn(tmp_path: Path):
    db_path = tmp_path / "tiny_feed.sqlite"
    build_tiny_feed_db(db_path)
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()


def test_load_feed_posts_returns_expected_shape(tiny_conn):
    df = load_feed_posts(tiny_conn)

    assert set(df.columns) >= {
        "id", "user_id", "body", "privacy",
        "likes_count", "comments_count", "created_at",
    }
    assert len(df) == 6
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])


def test_load_feed_posts_privacy_values_preserved(tiny_conn):
    df = load_feed_posts(tiny_conn)
    assert set(df["privacy"].unique()) == {"public", "friends"}


def test_load_users_min_returns_id_and_created_at(tiny_conn):
    df = load_users_min(tiny_conn)
    assert set(df.columns) >= {"id", "created_at"}
    assert len(df) == 4
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])


def test_load_friendships_only_returns_accepted_by_default(tiny_conn):
    df = load_friendships(tiny_conn)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "accepted"
    assert df.iloc[0]["requester_id"] == 1
    assert df.iloc[0]["receiver_id"] == 2


def test_load_friendships_can_include_pending(tiny_conn):
    df = load_friendships(tiny_conn, include_pending=True)
    assert len(df) == 2
    assert set(df["status"]) == {"accepted", "pending"}


def test_load_post_media_returns_rows_per_post(tiny_conn):
    df = load_post_media(tiny_conn)
    assert len(df) == 2
    assert set(df["media_type"]) == {"image", "video"}


def test_load_user_blocks_empty_by_default(tiny_conn):
    df = load_user_blocks(tiny_conn)
    assert len(df) == 0
    assert set(df.columns) >= {"blocker_id", "blocked_id"}


def test_load_user_mutes_empty_by_default(tiny_conn):
    df = load_user_mutes(tiny_conn)
    assert len(df) == 0
    assert set(df.columns) >= {"muter_id", "muted_id"}


def test_load_post_impressions_returns_all_rows(tiny_conn):
    from ml.feed_data_loader import load_post_impressions
    df = load_post_impressions(tiny_conn)
    assert set(df.columns) >= {
        "user_id", "feed_post_id", "served_at",
        "source", "position_in_feed", "viewed_for_ms",
    }
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["served_at"])


def test_load_post_impressions_filter_by_since(tiny_conn):
    from datetime import datetime
    from ml.feed_data_loader import load_post_impressions
    cutoff = datetime(2026, 7, 9, 12, 0, 0)  # keep only the last 24h impression (impression 3)
    df = load_post_impressions(tiny_conn, since=cutoff)
    assert len(df) == 1


def test_load_interactions_log_returns_all_rows(tiny_conn):
    from ml.feed_data_loader import load_interactions_log
    df = load_interactions_log(tiny_conn)
    assert set(df.columns) >= {
        "user_id", "feed_post_id", "interaction_type",
        "reaction_type", "occurred_at", "source",
    }
    assert len(df) == 4
    assert pd.api.types.is_datetime64_any_dtype(df["occurred_at"])


def test_load_interactions_log_filter_by_type(tiny_conn):
    from ml.feed_data_loader import load_interactions_log
    df = load_interactions_log(tiny_conn, interaction_types=("like",))
    assert len(df) == 3
    assert set(df["interaction_type"]) == {"like"}


def test_optional_telemetry_tables_return_empty_frames_when_not_migrated(tiny_conn):
    from ml.feed_data_loader import load_interactions_log, load_post_impressions

    tiny_conn.execute("DROP TABLE post_impressions")
    tiny_conn.execute("DROP TABLE feed_post_interactions_log")

    impressions = load_post_impressions(tiny_conn)
    interactions = load_interactions_log(tiny_conn)

    assert impressions.empty
    assert interactions.empty
    assert pd.api.types.is_datetime64_any_dtype(impressions["served_at"])
    assert pd.api.types.is_datetime64_any_dtype(interactions["occurred_at"])
