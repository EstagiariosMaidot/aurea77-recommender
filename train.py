"""Train the LTR recommender and write artifacts.

Usage (SQLite — local dev):
    python -m train \
        --db-path /path/to/database.sqlite \
        --out-dir artifacts

Usage (MySQL — production):
    DATABASE_URL=mysql://user:pass@host:3306/aurea77 python -m train \
        --out-dir artifacts
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from database import get_connection
from ml.artifacts import save_artifacts
from ml.data_loader import (
    load_athlete_profiles,
    load_events,
    load_registrations,
    load_reviews,
)
from ml.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    build_event_text,
    embed_sports_profile,
    embed_texts,
)
from ml.features import EVENT_TRAINING_SCHEMA_VERSION, FEATURE_NAMES, build_feature_row, bayesian_mean
from ml.labels import graded_relevance, sample_negatives
from ml.splits import temporal_cutoff, temporal_split


def _ro_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _user_semantic_profile(
    user_event_ids: set[int],
    event_id_to_row: dict[int, int],
    event_embeddings: np.ndarray,
    weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Weighted mean of the user's past-event embeddings; zero if empty."""
    rows = [event_id_to_row[eid] for eid in user_event_ids if eid in event_id_to_row]
    if not rows:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vecs = event_embeddings[rows]
    if weights:
        w = np.array(
            [weights.get(eid, 1.0) for eid in user_event_ids if eid in event_id_to_row],
            dtype=np.float32,
        )
        if w.sum() == 0:
            return vecs.mean(axis=0)
        return (vecs * w[:, None]).sum(axis=0) / w.sum()
    return vecs.mean(axis=0)


def _user_home_centroid(
    user_event_ids: set[int], events: pd.DataFrame
) -> tuple[float | None, float | None]:
    past = events.loc[events["id"].isin(user_event_ids), ["latitude", "longitude"]].dropna()
    if past.empty:
        return None, None
    return float(past["latitude"].mean()), float(past["longitude"].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=Path, default=None,
                    help="Path to SQLite file (ignored when DATABASE_URL is set)")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--cutoff-percentile", type=float, default=0.80)
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--hard-fraction", type=float, default=0.30)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # If --db-path is given (local dev with SQLite), temporarily set DATABASE_URL
    # so that get_connection() picks it up.  Otherwise DATABASE_URL must be set.
    if args.db_path is not None:
        os.environ["DATABASE_URL"] = f"sqlite:///{args.db_path.resolve()}"
        # Re-parse config (module-level _config was already cached)
        import database
        database._config = database._parse_database_url(os.environ["DATABASE_URL"])

    with get_connection() as conn:
        events = load_events(conn)
        registrations = load_registrations(conn)
        reviews = load_reviews(conn)
        profiles = load_athlete_profiles(conn)

    # ---------- Embeddings (once, over every event) ----------
    texts = build_event_text(events)
    event_embeddings = embed_texts(texts)
    event_ids = events["id"].tolist()
    event_id_to_row = {eid: i for i, eid in enumerate(event_ids)}

    # ---------- Temporal split ----------
    registrations = registrations.dropna(subset=["created_at"]).copy()
    cutoff = temporal_cutoff(registrations["created_at"], percentile=args.cutoff_percentile)
    train_regs, _test_regs = temporal_split(
        registrations, cutoff=cutoff, date_col="created_at"
    )
    # Edge case: all registrations share the same timestamp (e.g. tiny fixture).
    # Fall back to using all registrations for training so the pipeline doesn't crash.
    if train_regs.empty:
        train_regs = registrations.copy()

    # ---------- Labels ----------
    rel = graded_relevance(train_regs, reviews)

    # ---------- Negative sampling ----------
    negs = sample_negatives(
        registrations=train_regs,
        events=events,
        n_per_positive=args.neg_ratio,
        hard_fraction=args.hard_fraction,
        rng=rng,
    )

    # ---------- Assemble training set ----------
    positives = [
        {
            "user_id": int(row.user_id),
            "event_id": int(row.event_id),
            "relevance": rel[(row.user_id, row.event_id)],
            "created_at": row.created_at,
        }
        for row in train_regs.itertuples(index=False)
    ]
    all_pairs = pd.DataFrame(positives)
    if not negs.empty:
        all_pairs = pd.concat([all_pairs, negs], ignore_index=True)
    all_pairs = all_pairs.sort_values(["user_id", "created_at", "event_id"]).reset_index(drop=True)

    # Organizer ownership is static; all score/count aggregates below are
    # rebuilt as-of each query timestamp so a future interaction cannot leak
    # into a training feature.
    org_by_event = dict(zip(events["id"], events["organizer_id"]))
    profile_by_user = {r["user_id"]: r for _, r in profiles.iterrows()}

    # ---------- Build feature matrix ----------
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []

    for (uid, query_at), grp in all_pairs.groupby(["user_id", "created_at"], sort=True):
        query_now = pd.Timestamp(query_at).to_pydatetime()
        history_regs = train_regs[
            (train_regs["user_id"] == uid)
            & (train_regs["created_at"] < pd.Timestamp(query_at))
        ]
        history_event_ids = set(history_regs["event_id"].tolist())
        sports = tuple(sorted(profile_by_user.get(uid, {}).get("sports_practiced", []) or []))
        sports_emb = embed_sports_profile(sports)
        nationality = profile_by_user.get(uid, {}).get("nationality")

        home_lat, home_lng = _user_home_centroid(history_event_ids, events)
        user_emb = _user_semantic_profile(
            history_event_ids, event_id_to_row, event_embeddings
        )
        user_history_dates = sorted(
            events.loc[events["id"].isin(history_event_ids), "start_at"].dropna().tolist()
        )
        user_n = len(history_event_ids)

        historical_regs = train_regs[train_regs["created_at"] < pd.Timestamp(query_at)]
        event_popularity = historical_regs.groupby("event_id").size().to_dict()
        historical_reviews = reviews[reviews["created_at"] < pd.Timestamp(query_at)]
        event_score_sum = historical_reviews.groupby("event_id")["score"].sum().to_dict()
        event_score_cnt = historical_reviews.groupby("event_id")["score"].count().to_dict()
        org_score_sum: dict[int, float] = {}
        org_score_cnt: dict[int, int] = {}
        for review in historical_reviews.itertuples(index=False):
            org = org_by_event.get(review.event_id)
            if org is not None:
                org_score_sum[org] = org_score_sum.get(org, 0.0) + review.score
                org_score_cnt[org] = org_score_cnt.get(org, 0) + 1

        group_count = 0
        for row in grp.itertuples(index=False):
            eid = int(row.event_id)
            e = events.loc[events["id"] == eid]
            if e.empty:
                continue
            e = e.iloc[0]
            emb_row = event_embeddings[event_id_to_row[eid]]
            sem_sim = float(np.dot(user_emb, emb_row))
            sports_similarity = float(np.dot(sports_emb, emb_row)) if sports else 0.0
            org_id = org_by_event.get(eid)
            feat_row = build_feature_row(
                sem_sim=sem_sim,
                sports_similarity=sports_similarity,
                has_sports_practiced=bool(sports),
                user_home_lat=home_lat, user_home_lng=home_lng,
                user_nationality=nationality,
                user_history_dates=[pd.Timestamp(d).to_pydatetime() for d in user_history_dates],
                user_n_registrations=user_n,
                event_lat=e["latitude"], event_lng=e["longitude"],
                event_country_code=e["country_code"],
                event_start_at=pd.Timestamp(e["start_at"]).to_pydatetime(),
                query_now=query_now,
                event_popularity_raw=event_popularity.get(eid, 0),
                event_score_sum=event_score_sum.get(eid, 0.0),
                event_score_count=event_score_cnt.get(eid, 0),
                organizer_score_sum=org_score_sum.get(org_id, 0.0),
                organizer_score_count=org_score_cnt.get(org_id, 0),
            )
            rows.append(feat_row)
            labels.append(int(row.relevance))
            group_count += 1
        groups.append(group_count)

    X = np.vstack(rows)
    y = np.array(labels, dtype=np.int32)

    # ---------- Train ----------
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=args.seed,
    )
    model.fit(X, y, group=groups)

    # ---------- Save artifacts ----------
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(X.shape[0]),
        "n_users": int(all_pairs["user_id"].nunique()),
        "n_events": int(len(event_ids)),
        "cutoff": cutoff.isoformat(),
        "feature_names": FEATURE_NAMES,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "feature_schema_version": 2,
        "training_schema_version": EVENT_TRAINING_SCHEMA_VERSION,
        "model_version": datetime.now(timezone.utc).strftime("v1-%Y%m%dT%H%M%SZ"),
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "cutoff_percentile": args.cutoff_percentile,
            "neg_ratio": args.neg_ratio,
            "hard_fraction": args.hard_fraction,
            "seed": args.seed,
        },
    }
    # Write to a temp directory first, then atomic swap to avoid serving
    # partially-written artifacts during training.
    tmp_dir = Path(tempfile.mkdtemp(prefix="artifacts_tmp_", dir=args.out_dir.parent))
    save_artifacts(
        tmp_dir,
        model=model,
        event_embeddings=event_embeddings,
        event_ids=event_ids,
        metadata=metadata,
    )

    old_dir = args.out_dir.with_name(args.out_dir.name + "_old")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    if args.out_dir.exists():
        args.out_dir.rename(old_dir)
    tmp_dir.rename(args.out_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)

    print(json.dumps(
        {"status": "ok", "out_dir": str(args.out_dir), "metadata": metadata},
        indent=2,
    ))


if __name__ == "__main__":
    main()
