"""Full-stack smoke test: train → save → serve /posts/feed/{viewer}."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


def test_train_then_serve_end_to_end(tmp_path: Path, monkeypatch):
    """Train the feed model, then serve /posts/feed/{user_id} via TestClient.

    Confirms the full ML stack works:
    1. Builds a tiny deterministic DB
    2. Runs train_feed subprocess to train LGBMRanker and save artifacts
    3. Sets env vars and reloads FastAPI app
    4. Calls /health, /posts/embedded-ids, and /posts/feed/1
    5. Verifies structure and expected data

    This serves as a regression safety net for the full recommendation pipeline.
    """
    db_path = tmp_path / "tiny.sqlite"
    build_tiny_feed_db(db_path)
    artifacts = tmp_path / "artifacts" / "feed"

    # Train
    r = subprocess.run(
        [sys.executable, "-m", "train_feed",
         "--db-path", str(db_path),
         "--out-dir", str(artifacts),
         "--seed", "42",
         "--n-estimators", "10"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("FEED_ARTIFACTS_DIR", str(artifacts))
    import main
    importlib.reload(main)
    client = TestClient(main.app)

    # /health still works
    assert client.get("/health").status_code == 200
    # /posts/embedded-ids reports the trained ids
    body = client.get("/posts/embedded-ids").json()
    assert len(body["post_ids"]) > 0
    # /posts/feed/1 returns a payload with the expected structure
    body = client.get("/posts/feed/1?limit=5").json()
    assert body["user_id"] == 1
    assert isinstance(body["recommendations"], list)
