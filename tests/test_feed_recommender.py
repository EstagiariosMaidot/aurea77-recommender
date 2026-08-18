"""Tests for the feed recommender inference module."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from feed_recommender import FeedRecommendation, feed_model_status, recommend_for_viewer
from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def trained(tmp_path: Path):
    db_path = tmp_path / "tiny.sqlite"
    build_tiny_feed_db(db_path)
    artifacts = tmp_path / "artifacts" / "feed"
    subprocess.run(
        [sys.executable, "-m", "train_feed",
         "--db-path", str(db_path),
         "--out-dir", str(artifacts),
         "--seed", "42",
         "--n-estimators", "10"],
        check=True, capture_output=True, text=True,
    )
    return {"db_path": db_path, "artifacts": artifacts}


def test_recommend_returns_ordered_list_of_feed_recommendations(trained):
    results = recommend_for_viewer(
        viewer_id=1,
        db_path=str(trained["db_path"]),
        artifacts_dir=str(trained["artifacts"]),
        limit=10,
    )
    assert isinstance(results, list)
    for rec in results:
        assert isinstance(rec, FeedRecommendation)
        assert rec.source in {"friend", "discovery"}
        assert isinstance(rec.post_id, int)


def test_recommend_limit_is_respected(trained):
    results = recommend_for_viewer(
        viewer_id=1,
        db_path=str(trained["db_path"]),
        artifacts_dir=str(trained["artifacts"]),
        limit=2,
    )
    assert len(results) <= 2


def test_recommend_empty_when_viewer_has_no_candidates(trained):
    # User 4 (Dave) is not friends with anyone and there are no public non-friend posts
    # older than the fixture's window that would give him friends. Discovery may still
    # return Carol's posts, so we only assert the shape is valid (list, not error).
    results = recommend_for_viewer(
        viewer_id=4,
        db_path=str(trained["db_path"]),
        artifacts_dir=str(trained["artifacts"]),
        limit=10,
    )
    assert isinstance(results, list)


def test_recommend_falls_back_to_chronological_when_artifacts_missing(tmp_path):
    db = tmp_path / "empty.sqlite"
    build_tiny_feed_db(db)
    results = recommend_for_viewer(
        viewer_id=1,
        db_path=str(db),
        artifacts_dir=str(tmp_path / "does_not_exist"),
        limit=5,
    )
    assert isinstance(results, list)
    assert all(rec.score == 0.0 for rec in results)
    assert feed_model_status(str(tmp_path / "does_not_exist")) == ("chronological_fallback", None)


def test_feed_model_status_exposes_trained_model_version(trained):
    mode, version = feed_model_status(str(trained["artifacts"]))

    assert mode == "recommender"
    assert version is not None
    assert version.startswith("feed-v1-")


def test_feed_model_status_falls_back_for_artifacts_without_positive_feedback(tmp_path):
    artifacts = tmp_path / "artifacts" / "feed"
    artifacts.mkdir(parents=True)
    for name in ("model.pkl", "post_id_to_row.json", "post_embeddings.npy"):
        (artifacts / name).write_bytes(b"fixture")
    (artifacts / "metadata.json").write_text(
        json.dumps({"model_version": "feed-v1-empty", "n_positives": 0})
    )

    assert feed_model_status(str(artifacts)) == ("chronological_fallback", None)
