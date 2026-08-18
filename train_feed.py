"""Train the LightGBM ranker for the mixed post feed.

Usage
-----
    python -m train_feed [--db-path PATH] [--out-dir PATH] \\
        [--n-estimators N] [--neg-ratio X] [--friend-fraction F] \\
        [--seed N]

Reads posts, users, friendships, media, blocks, mutes, impressions and the
interactions log from ``DATABASE_URL`` (or ``--db-path`` for SQLite). Builds
positive and stratified-negative pairs, computes the feature matrix per pair,
trains an ``LGBMRanker`` with LambdaRank objective and saves the artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRanker

from ml.feed_candidate_pools import _friend_ids
from ml.feed_data_loader import (
    load_feed_posts,
    load_friendships,
    load_interactions_log,
    load_post_impressions,
    load_post_media,
    load_users_min,
)
from ml.feed_embeddings import embed_posts
from ml.feed_features import build_feature_row
from ml.feed_labels import (
    build_negative_pairs,
    build_positive_pairs,
    sample_negatives_stratified_by_source,
)
from ml.feed_utils import open_connection, viewer_profile_embedding


load_dotenv()


DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "feed"


def _build_dataset(
    *,
    posts: pd.DataFrame,
    users: pd.DataFrame,
    media: pd.DataFrame,
    friendships: pd.DataFrame,
    interactions: pd.DataFrame,
    impressions: pd.DataFrame,
    matrix: np.ndarray,
    id_to_row: dict[int, int],
    now: datetime,
    friend_fraction: float,
    neg_ratio: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Assemble the (X, y, group) matrices consumed by LGBMRanker."""
    positives = build_positive_pairs(interactions)
    negatives_all = build_negative_pairs(impressions, positives)

    # Target sample size: neg_ratio × number of positives.
    negatives = sample_negatives_stratified_by_source(
        negatives_all,
        friend_fraction=friend_fraction,
        total=len(positives) * neg_ratio,
        rng=rng,
    )

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    group_keys: list[int] = []

    def add_row(viewer_id: int, post_id: int, is_positive: int) -> None:
        author_id = int(posts[posts["id"] == post_id].iloc[0]["user_id"])
        f_ids = _friend_ids(viewer_id, friendships)
        viewer_positives = [pid for (uid, pid) in positives if uid == viewer_id]
        profile = viewer_profile_embedding(matrix, id_to_row, viewer_positives)
        row = build_feature_row(
            viewer_id=viewer_id,
            post_id=post_id,
            author_id=author_id,
            posts=posts,
            users=users,
            media=media,
            friendships=friendships,
            friend_ids=f_ids,
            interactions=interactions,
            impressions=impressions,
            viewer_profile_embedding=profile,
            post_embedding=matrix[id_to_row[post_id]],
            now=now,
        )
        rows.append(row)
        labels.append(is_positive)
        group_keys.append(viewer_id)

    for viewer_id, post_id in positives:
        if post_id in id_to_row:
            add_row(viewer_id, post_id, 1)

    for viewer_id, post_id in zip(negatives["user_id"].astype(int), negatives["feed_post_id"].astype(int)):
        if post_id in id_to_row:
            add_row(viewer_id, post_id, 0)

    if not rows:
        return pd.DataFrame(), np.array([], dtype=np.int32), np.array([], dtype=np.int64)

    # LightGBM ranker needs rows sorted by group and a group-count array
    # aligned with that order.
    sort_idx = np.argsort(group_keys, kind="stable")
    X = pd.DataFrame(rows).iloc[sort_idx].reset_index(drop=True)
    y = np.array(labels, dtype=np.int32)[sort_idx]
    sorted_groups = np.array(group_keys)[sort_idx]
    _, groups = np.unique(sorted_groups, return_counts=True)
    return X, y, groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--friend-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = open_connection(args.db_path)
    try:
        posts = load_feed_posts(conn)
        users = load_users_min(conn)
        friendships = load_friendships(conn)
        media = load_post_media(conn)
        interactions = load_interactions_log(conn)
        impressions = load_post_impressions(conn)
    finally:
        conn.close()

    version = "feed-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if posts.empty:
        # A newly deployed social feed may legitimately contain no posts yet.
        # Record an explicit no-data artifact so the retrain wrapper can keep
        # chronological serving without attempting embeddings or promotion.
        (out_dir / "metadata.json").write_text(
            json.dumps({
                "model_version": version,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_positives": 0,
                "n_negatives": 0,
                "embedding_dim": 0,
                "feature_names": [],
                "hyperparameters": {
                    "n_estimators": args.n_estimators,
                    "neg_ratio": args.neg_ratio,
                    "friend_fraction": args.friend_fraction,
                    "seed": args.seed,
                },
            }, indent=2)
        )
        print(f"No feed posts available; saved no-data metadata to {out_dir}")
        return 0

    matrix, id_to_row = embed_posts(posts)

    rng = np.random.default_rng(args.seed)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    X, y, groups = _build_dataset(
        posts=posts, users=users, media=media,
        friendships=friendships, interactions=interactions,
        impressions=impressions,
        matrix=matrix, id_to_row=id_to_row,
        now=now,
        friend_fraction=args.friend_fraction,
        neg_ratio=args.neg_ratio,
        rng=rng,
    )

    if len(X) == 0 or groups.sum() == 0:
        # Fixture case (no positives) — save a trivial model so evaluation can still run.
        model = None
    else:
        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=args.n_estimators,
            random_state=args.seed,
            verbose=-1,
        )
        model.fit(X, y, group=groups)

    joblib.dump(model, out_dir / "model.pkl")
    np.save(out_dir / "post_embeddings.npy", matrix)
    (out_dir / "post_id_to_row.json").write_text(
        json.dumps({str(k): v for k, v in id_to_row.items()})
    )
    (out_dir / "metadata.json").write_text(
        json.dumps({
            "model_version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_positives": int((y == 1).sum()) if len(y) else 0,
            "n_negatives": int((y == 0).sum()) if len(y) else 0,
            "embedding_dim": int(matrix.shape[1]),
            "feature_names": list(X.columns) if len(X) else [],
            "hyperparameters": {
                "n_estimators": args.n_estimators,
                "neg_ratio": args.neg_ratio,
                "friend_fraction": args.friend_fraction,
                "seed": args.seed,
            },
        }, indent=2)
    )
    print(f"Saved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
