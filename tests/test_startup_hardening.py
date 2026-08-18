"""Startup hardening — the recommender must refuse to start without RECOMMENDER_API_KEY.

Runs the import in a fresh subprocess so module state is isolated and the test
is deterministic whether run alone or as part of the full suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_PROJECT_DIR = Path(__file__).resolve().parent.parent


def _run_import_main(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "import main"],
        capture_output=True,
        text=True,
        cwd=_PROJECT_DIR,
        env=env,
    )


def test_app_refuses_to_start_without_api_key() -> None:
    env = {k: v for k, v in os.environ.items() if k != "RECOMMENDER_API_KEY"}
    result = _run_import_main(env)
    assert result.returncode != 0
    assert "RECOMMENDER_API_KEY" in result.stderr


def test_app_starts_with_api_key() -> None:
    env = {**os.environ, "RECOMMENDER_API_KEY": "secret-abc"}
    result = _run_import_main(env)
    assert result.returncode == 0, f"stderr: {result.stderr}"
