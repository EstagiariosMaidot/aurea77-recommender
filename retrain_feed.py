"""Retrain pipeline: train → evaluate → promote (or abort).

Trains a fresh model into a staging directory. If the quality gate in
``evaluate_feed`` passes, atomically swaps the staging into the live artifacts
directory, backing up the previous version. Otherwise, moves the failed
staging aside for inspection and leaves the live model untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).resolve().parent / "artifacts" / "feed"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and evaluate but never promote.")
    args = parser.parse_args()

    live = Path(args.out_dir)
    staging = live.parent / (live.name + "_staging")
    staging.mkdir(parents=True, exist_ok=True)

    train_cmd = [sys.executable, "-m", "train_feed",
                 "--out-dir", str(staging), "--n-estimators", "50"]
    if args.db_path:
        train_cmd.extend(["--db-path", args.db_path])
    r = subprocess.run(train_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"Retrain FAILED at train step: {r.stderr}\n")
        return 1

    metadata_path = staging / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("n_positives", 0)) <= 0:
        # A model fitted without positive interactions cannot provide a useful
        # ordering. Keep serving the built-in chronological fallback until the
        # API has collected enough impression/interactions telemetry.
        shutil.rmtree(staging)
        print("Retrain SKIPPED — no positive feed interactions available yet")
        return 0

    eval_cmd = [sys.executable, "-m", "evaluate_feed",
                "--artifacts-dir", str(staging)]
    if args.db_path:
        eval_cmd.extend(["--db-path", args.db_path])
    r = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(r.stdout)
    gate_passed = r.returncode == 0

    if args.dry_run:
        print("Retrain dry-run complete — staging kept at", staging)
        return 0 if gate_passed else 1

    if not gate_passed:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failed = live.parent / f"{live.name}_failed_{ts}"
        shutil.move(str(staging), str(failed))
        print(f"Retrain ABORTED — staging archived at {failed}")
        return 1

    if live.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = live.parent / f"{live.name}_backup_{ts}"
        shutil.move(str(live), str(backup))
        print(f"Previous model backed up to {backup}")

    shutil.move(str(staging), str(live))
    print(f"Retrain PROMOTED — live model at {live}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
