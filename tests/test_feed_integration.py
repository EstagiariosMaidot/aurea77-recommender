"""End-to-end smoke test: load → embed → build pools → build features."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.feed_data_loader import (
    load_feed_posts,
    load_users_min,
    load_friendships,
    load_post_media,
    load_user_blocks,
    load_user_mutes,
    load_interactions_log,
    load_post_impressions,
)
from ml.feed_embeddings import embed_posts
from ml.feed_candidate_pools import build_friend_pool, build_discovery_pool
from ml.feed_features import build_feature_row
from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "tiny.sqlite"
    build_tiny_feed_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        yield {
            "conn": conn,
            "now": datetime(2026, 7, 10, 12, 0, 0),
        }
    finally:
        conn.close()


@pytest.mark.parametrize(
    "viewer_id, viewer_name",
    [
        (1, "alice-cold-start"),   # Alice has no interactions in the fixture → zero profile
        (2, "bob-warm"),            # Bob has 2 like interactions → profile averages those embeddings
    ],
)
def test_end_to_end_pipeline_produces_feature_matrix(env, viewer_id, viewer_name):
    conn = env["conn"]
    now = env["now"]

    # Load
    posts = load_feed_posts(conn)
    users = load_users_min(conn)
    friendships = load_friendships(conn)
    media = load_post_media(conn)
    blocks = load_user_blocks(conn)
    mutes = load_user_mutes(conn)
    interactions = load_interactions_log(conn)
    impressions = load_post_impressions(conn)

    # Embed
    matrix, id_to_row = embed_posts(posts)
    assert matrix.shape[0] == len(posts)

    # Pools for the parametrised viewer
    friend_pool = build_friend_pool(
        viewer_id=viewer_id, posts=posts, friendships=friendships,
        blocks=blocks, mutes=mutes, now=now, window_days=14,
    )
    discovery_pool = build_discovery_pool(
        viewer_id=viewer_id, posts=posts, friendships=friendships,
        blocks=blocks, mutes=mutes, now=now, window_days=14,
        min_engagement=0,
    )

    # Combined candidate set (deduped)
    candidate_ids = list(dict.fromkeys(friend_pool + discovery_pool))
    assert len(candidate_ids) > 0

    # Viewer profile embedding: average of embeddings of posts the viewer interacted with.
    interacted_post_ids = interactions[interactions["user_id"] == viewer_id]["feed_post_id"].astype(int).tolist()
    if interacted_post_ids:
        rows = [id_to_row[pid] for pid in interacted_post_ids if pid in id_to_row]
        profile = matrix[rows].mean(axis=0) if rows else np.zeros(matrix.shape[1])
    else:
        # Cold user fallback: use a zero profile
        profile = np.zeros(matrix.shape[1])

    # Friend id set
    friend_ids = set(
        friendships[friendships["requester_id"] == viewer_id]["receiver_id"].tolist()
        + friendships[friendships["receiver_id"] == viewer_id]["requester_id"].tolist()
    )

    # Build feature rows
    feature_rows = []
    for post_id in candidate_ids:
        author_id = int(posts[posts["id"] == post_id].iloc[0]["user_id"])
        row = build_feature_row(
            viewer_id=viewer_id,
            post_id=post_id,
            author_id=author_id,
            posts=posts,
            users=users,
            media=media,
            friendships=friendships,
            friend_ids=friend_ids,
            interactions=interactions,
            impressions=impressions,
            viewer_profile_embedding=profile,
            post_embedding=matrix[id_to_row[post_id]],
            now=now,
        )
        feature_rows.append({"post_id": post_id, **row})

    df = pd.DataFrame(feature_rows)
    # Sanity: one row per candidate, all expected feature columns present.
    assert len(df) == len(candidate_ids)
    expected_cols = {
        "post_id", "has_media", "media_type_code", "body_length_bucket",
        "post_age_hours", "likes_normalized", "comments_normalized",
        "engagement_velocity", "is_friend", "friendship_age_days",
        "author_interaction_rate", "affinity_via_friends",
        "viewer_n_interactions", "viewer_n_friends", "viewer_days_since_signup",
        "sem_sim",
    }
    assert expected_cols.issubset(set(df.columns))
