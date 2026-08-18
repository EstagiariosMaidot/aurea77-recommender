"""Tests for feed feature functions (pure)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.feed_features import (
    feature_has_media,
    feature_media_type_code,
    feature_body_length_bucket,
    feature_post_age_hours,
    feature_likes_normalized,
    feature_comments_normalized,
    feature_engagement_velocity,
    feature_is_friend,
    feature_friendship_age_days,
    feature_author_interaction_rate,
    feature_affinity_via_friends,
    feature_viewer_n_interactions,
    feature_viewer_n_friends,
    feature_viewer_days_since_signup,
    feature_sem_sim,
    build_feature_row,
)


NOW = datetime(2026, 7, 10, 12, 0, 0)


def test_has_media_true_when_post_has_row_in_media():
    media = pd.DataFrame([{"post_id": 1, "media_type": "image", "sort_order": 0}])
    assert feature_has_media(post_id=1, media=media) == 1
    assert feature_has_media(post_id=2, media=media) == 0


def test_media_type_code():
    media = pd.DataFrame([
        {"post_id": 1, "media_type": "image", "sort_order": 0},
        {"post_id": 2, "media_type": "video", "sort_order": 0},
    ])
    assert feature_media_type_code(post_id=1, media=media) == 1
    assert feature_media_type_code(post_id=2, media=media) == 2
    assert feature_media_type_code(post_id=3, media=media) == 0


def test_body_length_bucket():
    assert feature_body_length_bucket("") == 0
    assert feature_body_length_bucket("Hi") == 0
    assert feature_body_length_bucket("A" * 100) == 1
    assert feature_body_length_bucket("A" * 500) == 2
    assert feature_body_length_bucket(None) == 0


def test_post_age_hours():
    created = NOW - timedelta(hours=5)
    assert feature_post_age_hours(created_at=created, now=NOW) == pytest.approx(5.0, rel=1e-3)


def test_post_age_hours_never_negative():
    # Future timestamps clamp to 0
    future = NOW + timedelta(hours=2)
    assert feature_post_age_hours(created_at=future, now=NOW) == 0.0


def test_likes_normalized_uses_days_since_creation():
    # 10 likes over 5 days → 2.0
    created = NOW - timedelta(days=5)
    assert feature_likes_normalized(likes_count=10, created_at=created, now=NOW) == pytest.approx(2.0, rel=1e-2)


def test_likes_normalized_for_new_post_returns_raw_count():
    # A post less than 1 day old shouldn't inflate; treat "age" as 1 day min.
    created = NOW - timedelta(hours=2)
    assert feature_likes_normalized(likes_count=5, created_at=created, now=NOW) == pytest.approx(5.0, rel=1e-2)


def test_comments_normalized_uses_days_since_creation():
    created = NOW - timedelta(days=4)
    assert feature_comments_normalized(comments_count=8, created_at=created, now=NOW) == pytest.approx(2.0, rel=1e-2)


def test_engagement_velocity_last_6h_window():
    interactions = pd.DataFrame([
        {"feed_post_id": 1, "interaction_type": "like", "occurred_at": NOW - timedelta(hours=1)},
        {"feed_post_id": 1, "interaction_type": "like", "occurred_at": NOW - timedelta(hours=3)},
        {"feed_post_id": 1, "interaction_type": "like", "occurred_at": NOW - timedelta(hours=10)},  # outside window
        {"feed_post_id": 2, "interaction_type": "like", "occurred_at": NOW - timedelta(hours=2)},
    ])
    assert feature_engagement_velocity(post_id=1, interactions=interactions, now=NOW) == 2
    assert feature_engagement_velocity(post_id=2, interactions=interactions, now=NOW) == 1
    assert feature_engagement_velocity(post_id=999, interactions=interactions, now=NOW) == 0


def test_is_friend_boolean():
    friend_ids = {2, 3}
    assert feature_is_friend(author_id=2, friend_ids=friend_ids) == 1
    assert feature_is_friend(author_id=4, friend_ids=friend_ids) == 0


def test_friendship_age_days_for_existing_friendship():
    friendships = pd.DataFrame([
        {"requester_id": 1, "receiver_id": 2, "status": "accepted",
         "created_at": pd.Timestamp(NOW - timedelta(days=150))},
    ])
    age = feature_friendship_age_days(
        viewer_id=1, author_id=2, friendships=friendships, now=NOW,
    )
    assert age == pytest.approx(150.0, rel=1e-2)


def test_friendship_age_days_returns_nan_when_not_friends():
    friendships = pd.DataFrame(columns=["requester_id", "receiver_id", "status", "created_at"])
    age = feature_friendship_age_days(
        viewer_id=1, author_id=2, friendships=friendships, now=NOW,
    )
    assert math.isnan(age)


def test_author_interaction_rate_uses_impressions_as_denominator():
    interactions = pd.DataFrame([
        {"user_id": 1, "feed_post_id": 10, "interaction_type": "like"},
        {"user_id": 1, "feed_post_id": 11, "interaction_type": "like"},
    ])
    impressions = pd.DataFrame([
        {"user_id": 1, "feed_post_id": 10},
        {"user_id": 1, "feed_post_id": 11},
        {"user_id": 1, "feed_post_id": 12},
        {"user_id": 1, "feed_post_id": 13},
    ])
    posts_by_author = pd.DataFrame([
        {"id": 10, "user_id": 5},
        {"id": 11, "user_id": 5},
        {"id": 12, "user_id": 5},
        {"id": 13, "user_id": 5},
    ])
    rate = feature_author_interaction_rate(
        viewer_id=1, author_id=5,
        interactions=interactions,
        impressions=impressions,
        posts=posts_by_author,
    )
    # 2 interactions / 4 impressions of author's posts = 0.5
    assert rate == pytest.approx(0.5, rel=1e-3)


def test_author_interaction_rate_zero_when_no_impressions():
    rate = feature_author_interaction_rate(
        viewer_id=1, author_id=99,
        interactions=pd.DataFrame(columns=["user_id", "feed_post_id", "interaction_type"]),
        impressions=pd.DataFrame(columns=["user_id", "feed_post_id"]),
        posts=pd.DataFrame(columns=["id", "user_id"]),
    )
    assert rate == 0.0


def test_affinity_via_friends_ratio():
    # 3 friends of viewer 1. Author 42. 2 of the 3 friends have interacted with author 42.
    friend_ids = {10, 20, 30}
    interactions = pd.DataFrame([
        {"user_id": 10, "feed_post_id": 100},
        {"user_id": 20, "feed_post_id": 100},
        # user 30 never interacted with any author-42 post
    ])
    posts = pd.DataFrame([
        {"id": 100, "user_id": 42},
    ])
    affinity = feature_affinity_via_friends(
        author_id=42,
        friend_ids=friend_ids,
        interactions=interactions,
        posts=posts,
    )
    assert affinity == pytest.approx(2 / 3, rel=1e-3)


def test_affinity_via_friends_zero_when_no_friends():
    aff = feature_affinity_via_friends(
        author_id=42,
        friend_ids=set(),
        interactions=pd.DataFrame(columns=["user_id", "feed_post_id"]),
        posts=pd.DataFrame(columns=["id", "user_id"]),
    )
    assert aff == 0.0


def test_viewer_n_interactions_counts_recent():
    interactions = pd.DataFrame([
        {"user_id": 1, "occurred_at": NOW - timedelta(days=10)},
        {"user_id": 1, "occurred_at": NOW - timedelta(days=95)},   # older than window
        {"user_id": 1, "occurred_at": NOW - timedelta(days=30)},
        {"user_id": 2, "occurred_at": NOW - timedelta(days=5)},
    ])
    assert feature_viewer_n_interactions(viewer_id=1, interactions=interactions, now=NOW, days=90) == 2
    assert feature_viewer_n_interactions(viewer_id=99, interactions=interactions, now=NOW, days=90) == 0


def test_viewer_n_friends():
    assert feature_viewer_n_friends({1, 2, 3}) == 3
    assert feature_viewer_n_friends(set()) == 0


def test_viewer_days_since_signup():
    users = pd.DataFrame([
        {"id": 1, "created_at": pd.Timestamp(NOW - timedelta(days=42))},
    ])
    assert feature_viewer_days_since_signup(viewer_id=1, users=users, now=NOW) == pytest.approx(42.0, rel=1e-2)


def test_viewer_days_since_signup_missing_user():
    users = pd.DataFrame(columns=["id", "created_at"])
    result = feature_viewer_days_since_signup(viewer_id=999, users=users, now=NOW)
    assert result != result  # NaN check


def test_sem_sim_cosine():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert feature_sem_sim(v1, v2) == pytest.approx(1.0)

    orthogonal = np.array([0.0, 1.0, 0.0])
    assert feature_sem_sim(v1, orthogonal) == pytest.approx(0.0)

    opposite = np.array([-1.0, 0.0, 0.0])
    assert feature_sem_sim(v1, opposite) == pytest.approx(-1.0)


def test_sem_sim_zero_vector_returns_zero():
    zero = np.zeros(3)
    other = np.array([1.0, 2.0, 3.0])
    assert feature_sem_sim(zero, other) == 0.0


def test_build_feature_row_contains_all_expected_keys():
    # Build a minimal valid input environment
    posts = pd.DataFrame([
        {"id": 100, "user_id": 42, "body": "Trail run.", "privacy": "public",
         "likes_count": 3, "comments_count": 1, "created_at": pd.Timestamp(NOW - timedelta(days=2))},
    ])
    users = pd.DataFrame([
        {"id": 1, "created_at": pd.Timestamp(NOW - timedelta(days=30))},
    ])
    row = build_feature_row(
        viewer_id=1,
        post_id=100,
        author_id=42,
        posts=posts,
        users=users,
        media=pd.DataFrame(columns=["post_id", "media_type", "sort_order"]),
        friendships=pd.DataFrame(columns=["requester_id", "receiver_id", "status", "created_at"]),
        friend_ids=set(),
        interactions=pd.DataFrame(columns=["user_id", "feed_post_id", "interaction_type", "occurred_at"]),
        impressions=pd.DataFrame(columns=["user_id", "feed_post_id"]),
        viewer_profile_embedding=np.ones(384) / np.sqrt(384),
        post_embedding=np.ones(384) / np.sqrt(384),
        now=NOW,
    )
    expected_keys = {
        "has_media", "media_type_code", "body_length_bucket",
        "post_age_hours", "likes_normalized", "comments_normalized",
        "engagement_velocity",
        "is_friend", "friendship_age_days",
        "author_interaction_rate", "affinity_via_friends",
        "viewer_n_interactions", "viewer_n_friends", "viewer_days_since_signup",
        "sem_sim",
    }
    assert set(row.keys()) >= expected_keys
