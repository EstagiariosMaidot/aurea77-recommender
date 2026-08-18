"""Tests for the feed training entrypoint.

The full training pipeline is exercised end-to-end against the tiny SQLite
fixture. We assert the artifacts land in the target directory with the
expected shape, not the numerical value of any weight.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def prepared(tmp_path: Path):
    db_path = tmp_path / "tiny.sqlite"
    build_tiny_feed_db(db_path)
    artifacts_dir = tmp_path / "artifacts" / "feed"
    return {"db_path": db_path, "artifacts_dir": artifacts_dir}


def test_train_feed_produces_the_expected_artifacts(prepared):
    result = subprocess.run(
        [
            sys.executable, "-m", "train_feed",
            "--db-path", str(prepared["db_path"]),
            "--out-dir", str(prepared["artifacts_dir"]),
            "--seed", "42",
            "--n-estimators", "20",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    d = prepared["artifacts_dir"]
    assert (d / "model.pkl").exists()
    assert (d / "post_embeddings.npy").exists()
    assert (d / "post_id_to_row.json").exists()

    metadata = json.loads((d / "metadata.json").read_text())
    assert metadata["embedding_dim"] == 384
    assert metadata["feature_names"], "metadata should list feature names"
    assert metadata["model_version"].startswith("feed-v1-")

    embeddings = np.load(d / "post_embeddings.npy")
    assert embeddings.shape[1] == 384
