"""HTTP tests for the feed FastAPI endpoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def app_with_trained_model(tmp_path: Path, monkeypatch):
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("FEED_ARTIFACTS_DIR", str(artifacts))

    import importlib
    import main
    importlib.reload(main)
    return TestClient(main.app)


def test_feed_endpoint_returns_recommendations(app_with_trained_model):
    r = app_with_trained_model.get("/posts/feed/1?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "recommendations" in body
    assert isinstance(body["recommendations"], list)
    assert body["ranking_mode"] == "recommender"
    assert body["model_version"].startswith("feed-v1-")
    for rec in body["recommendations"]:
        assert "post_id" in rec
        assert "source" in rec
        assert rec["source"] in {"friend", "discovery"}


def test_embedded_ids_endpoint(app_with_trained_model):
    r = app_with_trained_model.get("/posts/embedded-ids")
    assert r.status_code == 200
    body = r.json()
    assert "post_ids" in body
    assert isinstance(body["post_ids"], list)
    assert 1 in body["post_ids"]  # fixture has post id 1


def test_sync_endpoint_returns_ok(app_with_trained_model):
    r = app_with_trained_model.post("/posts/1/sync")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sync_endpoint_404_when_post_missing(app_with_trained_model):
    r = app_with_trained_model.post("/posts/99999/sync")
    assert r.status_code == 404


def test_delete_sync_endpoint(app_with_trained_model):
    r = app_with_trained_model.delete("/posts/1/sync")
    assert r.status_code == 200


def test_post_feed_accepts_bearer_jwt(app_with_trained_model, monkeypatch):
    monkeypatch.setenv("RECOMMENDER_JWT_SECRET", "test-jwt-secret")
    import time
    import jwt
    token = jwt.encode(
        {"user_id": 1, "exp": int(time.time()) + 300, "iat": int(time.time())},
        "test-jwt-secret",
        algorithm="HS256",
    )
    r = app_with_trained_model.post(
        "/posts/feed",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == 1
    assert isinstance(body["recommendations"], list)


def test_post_feed_rejects_missing_bearer(app_with_trained_model):
    r = app_with_trained_model.post("/posts/feed")
    assert r.status_code == 401


def test_post_feed_rejects_expired_bearer(app_with_trained_model, monkeypatch):
    monkeypatch.setenv("RECOMMENDER_JWT_SECRET", "test-jwt-secret")
    import time
    import jwt
    token = jwt.encode(
        {"user_id": 1, "exp": int(time.time()) - 10, "iat": int(time.time()) - 20},
        "test-jwt-secret",
        algorithm="HS256",
    )
    r = app_with_trained_model.post(
        "/posts/feed",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 401
