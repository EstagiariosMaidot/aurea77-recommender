"""Tests for feed candidate pool builders."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ml.feed_candidate_pools import build_friend_pool, build_discovery_pool
from ml.feed_data_loader import (
    load_feed_posts,
    load_friendships,
    load_user_blocks,
    load_user_mutes,
)
from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def loaded(tmp_path: Path):
    """Load all DataFrames from the tiny DB."""
    db_path = tmp_path / "tiny.sqlite"
    build_tiny_feed_db(db_path)
    conn = sqlite3.connect(str(db_path))
    result = {
        "posts": load_feed_posts(conn),
        "friendships": load_friendships(conn),
        "blocks": load_user_blocks(conn),
        "mutes": load_user_mutes(conn),
        "now": datetime(2026, 7, 10, 12, 0, 0),
    }
    conn.close()
    return result


def test_friend_pool_returns_posts_from_accepted_friends(loaded):
    # User 1 is friends (accepted) with user 2. Bob (user 2) authored posts 3 and 4.
    pool = build_friend_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
    )
    assert set(pool) == {3, 4}


def test_friend_pool_excludes_viewer_own_posts(loaded):
    # Viewer's own posts should NOT be in their friend pool.
    pool = build_friend_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
    )
    assert 1 not in pool and 2 not in pool


def test_friend_pool_respects_window_days(loaded):
    # window_days=1 keeps only very recent posts
    pool = build_friend_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=1,
    )
    # Bob's post 4 is 2 days old (outside window); Bob's post 3 is 4 days old (also outside).
    assert pool == []


def test_friend_pool_excludes_blocked_authors(loaded):
    # Add a block: viewer=1 blocked user 2
    blocks = pd.DataFrame([{"blocker_id": 1, "blocked_id": 2}])
    pool = build_friend_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=blocks,
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
    )
    assert pool == []


def test_friend_pool_returns_empty_for_user_with_no_friends(loaded):
    pool = build_friend_pool(
        viewer_id=4,   # dave has no friendships in the tiny DB
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
    )
    assert pool == []


def test_discovery_pool_excludes_friends_and_self(loaded):
    # Viewer 1: friends with 2. Discovery pool should never include posts from 1 or 2.
    pool = build_discovery_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
        min_engagement=0,
    )
    # Carol (id=3) posts 5 and 6 are candidates; post 6 is 'friends' privacy so out.
    assert 5 in pool
    assert 6 not in pool  # non-public privacy excluded
    assert 1 not in pool and 2 not in pool  # own + friends


def test_discovery_pool_filters_by_min_engagement(loaded):
    # min_engagement=1 keeps posts with likes+comments >= 1
    pool = build_discovery_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
        min_engagement=1,
    )
    # Carol's post 5 has 0 likes and 0 comments → excluded
    assert 5 not in pool


def test_discovery_pool_respects_window(loaded):
    # window_days=0 (only today) → nothing
    pool = build_discovery_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=0,
        min_engagement=0,
    )
    assert pool == []


def test_discovery_pool_excludes_blocked_authors(loaded):
    blocks = pd.DataFrame([{"blocker_id": 1, "blocked_id": 3}])
    pool = build_discovery_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=blocks,
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
        min_engagement=0,
    )
    assert pool == []


# ---------- Coverage additions (follow-ups from final review) ----------


def test_friend_pool_includes_members_privacy_posts(loaded):
    """`members` posts from friends should be visible in the friend pool.

    The fixture only seeds `public` and `friends`, so we inject a `members` post
    inline. Bob (user 2) is Alice's (user 1) friend; give him one members post.
    """
    posts_with_members = pd.concat(
        [
            loaded["posts"],
            pd.DataFrame([{
                "id": 100,
                "user_id": 2,  # Bob (friend of viewer 1)
                "body": "Bob's members-only post.",
                "privacy": "members",
                "likes_count": 0,
                "comments_count": 0,
                "created_at": pd.Timestamp(loaded["now"]) - pd.Timedelta(days=1),
                "updated_at": pd.NaT,
            }]),
        ],
        ignore_index=True,
    )

    pool = build_friend_pool(
        viewer_id=1,
        posts=posts_with_members,
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=loaded["mutes"],
        now=loaded["now"],
        window_days=14,
    )
    assert 100 in pool


def test_friend_pool_excludes_muted_authors(loaded):
    """Muted authors should be excluded from the friend pool, mirroring blocks."""
    mutes = pd.DataFrame([{"muter_id": 1, "muted_id": 2}])
    pool = build_friend_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=mutes,
        now=loaded["now"],
        window_days=14,
    )
    assert pool == []


def test_discovery_pool_excludes_muted_authors(loaded):
    """Muted authors should be excluded from the discovery pool, mirroring blocks."""
    mutes = pd.DataFrame([{"muter_id": 1, "muted_id": 3}])
    pool = build_discovery_pool(
        viewer_id=1,
        posts=loaded["posts"],
        friendships=loaded["friendships"],
        blocks=loaded["blocks"],
        mutes=mutes,
        now=loaded["now"],
        window_days=14,
        min_engagement=0,
    )
    assert pool == []
