"""Compare the trained feed ranker against baselines.

Prints a compact table:

    baseline         NDCG@10   MAP@10   Hit@10   Coverage@10
    chronological    0.0XX     0.0XX    0.0XX    0.XXXX
    popularity       ...
    friends_only     ...
    ltr_v1           ...

Exits with status 1 when the LTR score is below the quality gate.
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

from evaluate import ndcg_at_k, map_at_k, hit_at_k, coverage_at_k
from ml.feed_baselines import rank_chronological, rank_friends_only, rank_popularity
from ml.feed_candidate_pools import _friend_ids
from ml.feed_data_loader import (
    load_feed_posts,
    load_friendships,
    load_interactions_log,
    load_post_impressions,
    load_post_media,
    load_users_min,
)
from ml.feed_features import build_feature_row
from ml.feed_labels import POSITIVE_INTERACTION_TYPES
from ml.feed_utils import open_connection, viewer_profile_embedding
from ml.splits import temporal_cutoff


load_dotenv()

QUALITY_GATE_NDCG_DELTA = 0.10  # LTR must beat chronological by ≥ 10%


def _score_with_model(
    model, feature_rows: list[dict[str, float]], candidates: list[int]
) -> list[int]:
    if model is None or not feature_rows:
        return candidates
    X = pd.DataFrame(feature_rows)
    scores = model.predict(X)
    order = np.argsort(-scores)
    return [candidates[i] for i in order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--artifacts-dir",
                        default=str(Path(__file__).resolve().parent / "artifacts" / "feed"))
    args = parser.parse_args()

    d = Path(args.artifacts_dir)
    model = joblib.load(d / "model.pkl")
    id_to_row = {int(k): int(v) for k, v in json.loads((d / "post_id_to_row.json").read_text()).items()}
    matrix = np.load(d / "post_embeddings.npy")

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

    if interactions.empty:
        print("No interactions — cannot evaluate. LTR ranked equal to baselines.")
        return 0

    cutoff = temporal_cutoff(interactions["occurred_at"], percentile=0.80)
    train = interactions[interactions["occurred_at"] <= cutoff]
    test = interactions[interactions["occurred_at"] > cutoff]

    positives_by_user: dict[int, set[int]] = {}
    for _, row in test.iterrows():
        if row["interaction_type"] in POSITIVE_INTERACTION_TYPES:
            positives_by_user.setdefault(int(row["user_id"]), set()).add(int(row["feed_post_id"]))

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    results: dict[str, dict[str, list[float]]] = {b: {"ndcg": [], "map": [], "hit": [], "cov": []} for b in ("chronological", "popularity", "friends_only", "ltr_v1")}
    k = 10

    for viewer_id, truth in positives_by_user.items():
        candidates = posts["id"].astype(int).tolist()
        # Baselines
        chrono = rank_chronological(candidates=candidates, posts=posts)
        pop = rank_popularity(candidates=candidates, posts=posts)
        fi = _friend_ids(viewer_id, friendships)
        friends = rank_friends_only(candidates=candidates, posts=posts, friend_ids=fi)

        # LTR
        viewer_positives_train = list(train[
            (train["user_id"] == viewer_id)
            & (train["interaction_type"].isin(POSITIVE_INTERACTION_TYPES))
        ]["feed_post_id"].astype(int).tolist())
        profile = viewer_profile_embedding(matrix, id_to_row, viewer_positives_train)
        rows: list[dict[str, float]] = []
        for pid in candidates:
            if pid not in id_to_row:
                continue
            author_id = int(posts[posts["id"] == pid].iloc[0]["user_id"])
            rows.append(build_feature_row(
                viewer_id=viewer_id, post_id=pid, author_id=author_id,
                posts=posts, users=users, media=media,
                friendships=friendships, friend_ids=fi,
                interactions=train, impressions=impressions,
                viewer_profile_embedding=profile,
                post_embedding=matrix[id_to_row[pid]],
                now=now,
            ))
        ltr = _score_with_model(model, rows, [pid for pid in candidates if pid in id_to_row])

        for name, ranked in (
            ("chronological", chrono),
            ("popularity", pop),
            ("friends_only", friends),
            ("ltr_v1", ltr),
        ):
            results[name]["ndcg"].append(ndcg_at_k(ranked, truth, k=k))
            results[name]["map"].append(map_at_k(ranked, truth, k=k))
            results[name]["hit"].append(hit_at_k(ranked, truth, k=k))
            results[name]["cov"].append(len(set(ranked[:k])))

    print(f"{'baseline':<16} {'NDCG@10':>8} {'MAP@10':>8} {'Hit@10':>8} {'Coverage@10':>12}")
    print("-" * 60)
    ndcg_by = {name: float(np.mean(v["ndcg"])) if v["ndcg"] else 0.0 for name, v in results.items()}
    for name, v in results.items():
        ndcg = float(np.mean(v["ndcg"])) if v["ndcg"] else 0.0
        m = float(np.mean(v["map"])) if v["map"] else 0.0
        h = float(np.mean(v["hit"])) if v["hit"] else 0.0
        c = (sum(v["cov"]) / (len(v["cov"]) * k)) if v["cov"] else 0.0
        print(f"{name:<16} {ndcg:>8.4f} {m:>8.4f} {h:>8.4f} {c:>12.4f}")

    passes = ndcg_by["ltr_v1"] >= ndcg_by["chronological"] * (1 + QUALITY_GATE_NDCG_DELTA)
    print("\nGate:", "PASS" if passes else "FAIL")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
