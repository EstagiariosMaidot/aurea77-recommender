# tests/test_recommender.py
import sqlite3
from pathlib import Path

from recommender import Recommendation, recommend


def test_ltr_returns_top_n_with_known_fixture(tmp_path, tiny_db_path):
    import subprocess, sys
    out = tmp_path / "artifacts"
    subprocess.check_call([
        sys.executable, "-m", "train",
        "--db-path", str(tiny_db_path),
        "--out-dir", str(out),
        "--n-estimators", "50",
        "--seed", "42",
    ])

    uri = f"file:{tiny_db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        recs = recommend(conn, user_id=1, limit=5, artifacts_dir=out)
    finally:
        conn.close()

    assert 1 <= len(recs) <= 5
    assert all(isinstance(r, Recommendation) for r in recs)
    assert all(-5.0 < r.score < 5.0 for r in recs)


def test_cold_start_user_returns_nonempty_via_profile_features(tmp_path, tiny_db_path):
    out = tmp_path / "artifacts"
    import subprocess, sys
    subprocess.check_call([
        sys.executable, "-m", "train",
        "--db-path", str(tiny_db_path), "--out-dir", str(out),
        "--n-estimators", "50", "--seed", "42",
    ])

    conn = sqlite3.connect(f"file:{tiny_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        recs = recommend(conn, user_id=5, limit=5, artifacts_dir=out)
    finally:
        conn.close()

    assert len(recs) >= 1


def test_falls_back_to_chronological_when_artifacts_missing(tmp_path, tiny_db_path):
    conn = sqlite3.connect(f"file:{tiny_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        recs = recommend(
            conn, user_id=1, limit=5, artifacts_dir=tmp_path / "does-not-exist"
        )
    finally:
        conn.close()

    assert recs
    assert all(rec.score == 0.0 for rec in recs)
