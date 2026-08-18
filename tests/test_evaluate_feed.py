"""Smoke test for the feed evaluation entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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


def test_evaluate_feed_prints_metrics(trained):
    result = subprocess.run(
        [sys.executable, "-m", "evaluate_feed",
         "--db-path", str(trained["db_path"]),
         "--artifacts-dir", str(trained["artifacts"])],
        capture_output=True, text=True,
    )
    # Fixture is tiny → gate may fail. We only assert the report was printed.
    assert "NDCG@10" in result.stdout
    assert "chronological" in result.stdout
    assert "popularity" in result.stdout
    assert "friends_only" in result.stdout
    assert "ltr_v1" in result.stdout
