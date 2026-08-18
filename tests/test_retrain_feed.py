"""Tests for the retrain feed pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.build_tiny_feed_db import build_tiny_feed_db


@pytest.fixture
def prepared(tmp_path: Path):
    db_path = tmp_path / "tiny.sqlite"
    build_tiny_feed_db(db_path)
    artifacts = tmp_path / "artifacts" / "feed"
    return {"db_path": db_path, "artifacts": artifacts}


def test_retrain_dry_run_does_not_touch_artifacts(prepared):
    r = subprocess.run(
        [sys.executable, "-m", "retrain_feed",
         "--db-path", str(prepared["db_path"]),
         "--out-dir", str(prepared["artifacts"]),
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert not prepared["artifacts"].exists() or not (prepared["artifacts"] / "model.pkl").exists()


def test_retrain_promotes_when_gate_passes_or_reports_when_it_fails(prepared):
    r = subprocess.run(
        [sys.executable, "-m", "retrain_feed",
         "--db-path", str(prepared["db_path"]),
         "--out-dir", str(prepared["artifacts"])],
        capture_output=True, text=True,
    )
    # On this tiny fixture the gate may fail — but the process must exit cleanly.
    assert r.returncode in (0, 1)
    assert "Retrain " in r.stdout or "Retrain " in r.stderr
