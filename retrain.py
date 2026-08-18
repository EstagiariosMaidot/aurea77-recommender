"""Automated retrain pipeline: train → evaluate → swap (or abort).

Trains a new model to a staging directory, runs the evaluation quality gate,
and only promotes the new model to production if the gate passes.  If it
fails, the current production model is left untouched.

Usage (local dev — SQLite):
    python -m retrain --db-path ../aurea77-api/database/database.sqlite

Usage (production — MySQL, via cron):
    DATABASE_URL=mysql://user:pass@host:3306/aurea77 python -m retrain

Cron example (every Sunday at 03:00):
    0 3 * * 0  cd /opt/aurea77-recommendation && /opt/aurea77-recommendation/.venv/bin/python -m retrain >> /var/log/aurea77-retrain.log 2>&1
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# Directories
PRODUCTION_DIR = SCRIPT_DIR / "artifacts"
STAGING_DIR = SCRIPT_DIR / "artifacts_staging"
BACKUP_DIR = SCRIPT_DIR / "artifacts_backup"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, streaming output to stdout."""
    return subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Automated retrain pipeline")
    ap.add_argument("--db-path", type=Path, default=None,
                    help="Path to SQLite file (ignored when DATABASE_URL is set)")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Train and evaluate but don't promote the model")
    args = ap.parse_args()

    _log("=== Retrain pipeline started ===")

    # ── Step 1: Clean up any leftover staging dir ─────────────────────
    if STAGING_DIR.exists():
        _log(f"Removing leftover staging dir: {STAGING_DIR}")
        shutil.rmtree(STAGING_DIR)

    # ── Step 2: Train to staging directory ────────────────────────────
    _log("Step 1/3: Training new model...")
    train_cmd = [
        PYTHON, "-m", "train",
        "--out-dir", str(STAGING_DIR),
        "--n-estimators", str(args.n_estimators),
        "--seed", str(args.seed),
    ]
    if args.db_path is not None:
        train_cmd.extend(["--db-path", str(args.db_path.resolve())])

    result = _run(train_cmd)
    if result.returncode != 0:
        _log("ABORT: Training failed.")
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        return 1

    # Verify staging artifacts were created
    if not (STAGING_DIR / "metadata.json").exists():
        _log("ABORT: Training completed but artifacts not found in staging dir.")
        return 1

    _log("Training complete.")

    # ── Step 3: Evaluate staged model ─────────────────────────────────
    _log("Step 2/3: Evaluating new model...")
    eval_cmd = [
        PYTHON, "-m", "evaluate",
        "--artifacts-dir", str(STAGING_DIR),
    ]
    if args.db_path is not None:
        eval_cmd.extend(["--db-path", str(args.db_path.resolve())])

    result = _run(eval_cmd)

    if result.returncode != 0:
        _log("ABORT: Quality gate FAILED. Keeping current production model.")
        # Keep staging for inspection, but rename it
        failed_dir = SCRIPT_DIR / f"artifacts_failed_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        STAGING_DIR.rename(failed_dir)
        _log(f"Failed model saved to {failed_dir.name} for inspection.")
        return 1

    _log("Quality gate PASSED.")

    # Log evaluation results
    eval_json = STAGING_DIR / "evaluation.json"
    if eval_json.exists():
        results = json.loads(eval_json.read_text())
        ltr = results.get("ltr_v1", {})
        _log(f"  NDCG@10={ltr.get('NDCG@10', '?'):.4f}  "
             f"Hit@10={ltr.get('Hit@10', '?'):.4f}  "
             f"Coverage@10={ltr.get('Coverage@10', '?'):.4f}")

    # ── Step 4: Promote to production (atomic swap) ───────────────────
    if args.dry_run:
        _log("DRY RUN: Skipping promotion. Cleaning up staging.")
        shutil.rmtree(STAGING_DIR)
        return 0

    _log("Step 3/3: Promoting new model to production...")

    # Backup current production model
    if PRODUCTION_DIR.exists():
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        PRODUCTION_DIR.rename(BACKUP_DIR)
        _log(f"Previous model backed up to {BACKUP_DIR.name}/")

    # Promote staging → production
    STAGING_DIR.rename(PRODUCTION_DIR)

    # Read new model version
    meta = json.loads((PRODUCTION_DIR / "metadata.json").read_text())
    _log(f"New model live: version={meta.get('model_version')}  "
         f"trained_at={meta.get('trained_at')}  "
         f"events={meta.get('n_events')}  "
         f"users={meta.get('n_users')}")

    _log("=== Retrain pipeline finished successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
