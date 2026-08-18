"""Tests for feed label building and sampling."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.feed_labels import (
    POSITIVE_INTERACTION_TYPES,
    build_positive_pairs,
    build_negative_pairs,
    sample_negatives_stratified_by_source,
)


NOW = datetime(2026, 7, 10, 12, 0, 0)


def test_positive_types_are_the_expected_four():
    assert POSITIVE_INTERACTION_TYPES == {"like", "comment", "share", "follow_from_post"}


def test_build_positive_pairs_extracts_positive_types_only():
    interactions = pd.DataFrame([
        {"user_id": 1, "feed_post_id": 10, "interaction_type": "like",   "occurred_at": NOW - timedelta(days=1)},
        {"user_id": 1, "feed_post_id": 11, "interaction_type": "comment","occurred_at": NOW - timedelta(days=1)},
        {"user_id": 1, "feed_post_id": 12, "interaction_type": "hide",   "occurred_at": NOW - timedelta(days=1)},
        {"user_id": 1, "feed_post_id": 13, "interaction_type": "report", "occurred_at": NOW - timedelta(days=1)},
        {"user_id": 2, "feed_post_id": 10, "interaction_type": "share",  "occurred_at": NOW - timedelta(days=1)},
    ])
    positives = build_positive_pairs(interactions)
    assert positives == {(1, 10), (1, 11), (2, 10)}


def test_build_negative_pairs_excludes_positives():
    impressions = pd.DataFrame([
        {"user_id": 1, "feed_post_id": 10, "source": "friend"},
        {"user_id": 1, "feed_post_id": 11, "source": "friend"},
        {"user_id": 1, "feed_post_id": 12, "source": "discovery"},
        {"user_id": 1, "feed_post_id": 13, "source": "discovery"},
    ])
    positives = {(1, 10), (1, 11)}
    negatives = build_negative_pairs(impressions, positives)
    # Only impressions not in the positive set survive.
    assert set(zip(negatives["user_id"], negatives["feed_post_id"])) == {(1, 12), (1, 13)}
    # The negative frame keeps `source` so downstream sampling can be stratified.
    assert set(negatives["source"]) == {"discovery"}


def test_sample_negatives_stratified_by_source_reaches_target_ratio():
    rng = np.random.default_rng(42)
    negatives = pd.DataFrame([
        {"user_id": 1, "feed_post_id": p, "source": "friend"}
        for p in range(100)
    ] + [
        {"user_id": 1, "feed_post_id": 200 + p, "source": "discovery"}
        for p in range(100)
    ])
    sampled = sample_negatives_stratified_by_source(
        negatives, friend_fraction=0.6, total=50, rng=rng,
    )
    # 60% of 50 = 30 friend negatives, 40% = 20 discovery negatives.
    counts = sampled["source"].value_counts().to_dict()
    assert counts["friend"] == 30
    assert counts["discovery"] == 20


def test_sample_negatives_returns_all_when_pool_smaller_than_target():
    rng = np.random.default_rng(0)
    negatives = pd.DataFrame([
        {"user_id": 1, "feed_post_id": 1, "source": "friend"},
        {"user_id": 1, "feed_post_id": 2, "source": "friend"},
    ])
    sampled = sample_negatives_stratified_by_source(
        negatives, friend_fraction=0.5, total=10, rng=rng,
    )
    # Only 2 friends available and no discovery — sampler returns everything available.
    assert len(sampled) == 2
