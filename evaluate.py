"""Offline evaluation: metrics, baselines, and quality gate."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import math
from typing import Iterable


def ndcg_at_k(ranked: list[int], truth: set[int], *, k: int) -> float:
    ranked_k = ranked[:k]
    if not truth:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked_k):
        if item in truth:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(ranked: list[int], truth: set[int], *, k: int) -> float:
    if not truth:
        return 0.0
    ranked_k = ranked[:k]
    hits = 0
    precisions = []
    for i, item in enumerate(ranked_k, start=1):
        if item in truth:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / min(len(truth), k)


def hit_at_k(ranked: list[int], truth: set[int], *, k: int) -> int:
    return 1 if any(item in truth for item in ranked[:k]) else 0


def coverage_at_k(
    top_k_per_user: dict[int, list[int]], catalog: Iterable[int], *, k: int
) -> float:
    cat_set = set(catalog)
    if not cat_set:
        return 0.0
    union: set[int] = set()
    for ranked in top_k_per_user.values():
        union.update(ranked[:k])
    return len(union & cat_set) / len(cat_set)


import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from database import get_connection
from ml.artifacts import ArtifactStore
from ml.data_loader import load_athlete_profiles, load_events, load_registrations, load_reviews
from ml.embeddings import embed_sports_profile, embed_texts, build_event_text
from ml.splits import temporal_cutoff


def _ro_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _candidate_pool(
    events: pd.DataFrame,
    *,
    registered_event_ids: set[int],
    query_at: pd.Timestamp,
) -> list[int]:
    """Events the user could actually have been shown at ``query_at``."""
    candidates = events[
        events["published_at"].notna()
        & (events["published_at"] <= query_at)
        & (events["start_at"] > query_at)
        & ~events["id"].isin(registered_event_ids)
    ]
    return candidates["id"].astype(int).tolist()


# ---------- Baselines ----------

def rank_random(candidates: list[int], rng: np.random.Generator) -> list[int]:
    order = list(candidates)
    rng.shuffle(order)
    return order


def rank_popularity(
    candidates: list[int], popularity: dict[int, int]
) -> list[int]:
    return sorted(candidates, key=lambda e: -popularity.get(e, 0))


def rank_sports(
    candidates: list[int],
    user_sports: tuple[str, ...], event_embeddings: np.ndarray,
    event_id_to_row: dict[int, int],
) -> list[int]:
    """Sports-profile semantic baseline on the provided candidate pool."""
    if not user_sports:
        return candidates
    user_vec = embed_sports_profile(user_sports)
    scored: list[tuple[int, float]] = []
    for eid in candidates:
        row = event_id_to_row.get(eid)
        if row is None:
            scored.append((eid, 0.0))
            continue
        scored.append((eid, float(np.dot(user_vec, event_embeddings[row]))))

    scored.sort(key=lambda p: -p[1])
    return [eid for eid, _ in scored]


def rank_ltr(
    conn, user_id: int, candidates: list[int], limit: int,
    artifacts_dir: Path, query_now=None, *, cached_store=None,
    preloaded_data=None,
) -> list[int]:
    from recommender import recommend
    recs = recommend(conn, user_id=user_id, limit=len(candidates),
                     artifacts_dir=artifacts_dir, query_now=query_now,
                     cached_store=cached_store, preloaded_data=preloaded_data)
    ordered = [r.event_id for r in recs if r.event_id in set(candidates)]
    missing = [c for c in candidates if c not in ordered]
    return (ordered + missing)[:limit * 20]


# ---------- Driver ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=Path, default=None,
                    help="Path to SQLite file (ignored when DATABASE_URL is set)")
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--cutoff-percentile", type=float, default=0.80)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-ndcg-vs-sports", "--min-ndcg-vs-cosine",
        dest="min_ndcg_vs_sports", type=float, default=0.0,
        help="Exit non-zero if LTR NDCG@k <= sports-profile NDCG@k + this margin.",
    )
    ap.add_argument("--min-coverage", type=float, default=0.20)
    args = ap.parse_args()

    # If --db-path is given, set DATABASE_URL for the connection module
    if args.db_path is not None:
        os.environ["DATABASE_URL"] = f"sqlite:///{args.db_path.resolve()}"
        import database
        database._config = database._parse_database_url(os.environ["DATABASE_URL"])

    rng = np.random.default_rng(args.seed)

    with get_connection() as conn:
        events = load_events(conn)
        registrations = load_registrations(conn)
        reviews = load_reviews(conn)
        registrations = registrations.dropna(subset=["created_at"]).copy()
        cutoff = temporal_cutoff(registrations["created_at"],
                                 percentile=args.cutoff_percentile)
        test_regs = registrations[registrations["created_at"] >= cutoff]

        # Sports are the explicit cold-start preference signal; categories are
        # intentionally absent from both the baseline and the learned model.
        profiles = load_athlete_profiles(conn)
        sports_by_user = {
            int(row.user_id): tuple(sorted(row.sports_practiced or []))
            for row in profiles.itertuples(index=False)
        }
        event_embeddings = embed_texts(build_event_text(events))
        event_id_to_row = {int(event_id): i for i, event_id in enumerate(events["id"])}
        cached_store = ArtifactStore.load(args.artifacts_dir)
        preloaded_data = (events, registrations, reviews, profiles)

        # One query per user and interaction timestamp. Every query has its
        # own historical registrations, available candidates and popularity
        # counts, which prevents evaluating with information from the future.
        queries: list[dict] = []
        for (uid, query_at), same_time_regs in test_regs.groupby(["user_id", "created_at"], sort=True):
            history = registrations[
                (registrations["user_id"] == uid)
                & (registrations["created_at"] < query_at)
            ]
            candidates = _candidate_pool(
                events,
                registered_event_ids=set(history["event_id"].tolist()),
                query_at=pd.Timestamp(query_at),
            )
            truth = set(same_time_regs["event_id"].astype(int).tolist()) & set(candidates)
            if not truth:
                continue
            queries.append({
                "id": (int(uid), pd.Timestamp(query_at).isoformat()),
                "user_id": int(uid),
                "query_at": pd.Timestamp(query_at),
                "candidates": candidates,
                "truth": truth,
                "history_count": int(len(history)),
            })
        evaluation_catalog = sorted({
            event_id for query in queries for event_id in query["candidates"]
        })

        results: dict[str, dict[str, float]] = {}
        baseline_names = ["random", "popularity", "sports_v0", "ltr_v1"]
        top_k_per_baseline: dict[str, dict[tuple[int, str], list[int]]] = {
            name: {} for name in baseline_names
        }

        if not queries:
            print("No valid test queries found after the temporal split.")
            print("Gate: SKIP")
            return 0

        for name in baseline_names:
            ndcg_scores, map_scores, hits = [], [], []
            for query in queries:
                query_at = query["query_at"]
                candidates = query["candidates"]
                if name == "random":
                    ranked = rank_random(candidates, rng)
                elif name == "popularity":
                    popularity = registrations[
                        registrations["created_at"] < query_at
                    ].groupby("event_id").size().to_dict()
                    ranked = rank_popularity(candidates, popularity)
                elif name == "sports_v0":
                    ranked = rank_sports(
                        candidates,
                        sports_by_user.get(query["user_id"], ()),
                        event_embeddings,
                        event_id_to_row,
                    )
                else:
                    ranked = rank_ltr(
                        conn,
                        query["user_id"],
                        candidates,
                        args.k,
                        args.artifacts_dir,
                        query_now=query_at.to_pydatetime(),
                        cached_store=cached_store,
                        preloaded_data=preloaded_data,
                    )
                top_k_per_baseline[name][query["id"]] = ranked[:args.k]
                ndcg_scores.append(ndcg_at_k(ranked, query["truth"], k=args.k))
                map_scores.append(map_at_k(ranked, query["truth"], k=args.k))
                hits.append(hit_at_k(ranked, query["truth"], k=args.k))
            results[name] = {
                f"NDCG@{args.k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
                f"MAP@{args.k}":  float(np.mean(map_scores))  if map_scores else 0.0,
                f"Hit@{args.k}":  float(np.mean(hits))        if hits else 0.0,
                f"Coverage@{args.k}": coverage_at_k(
                    top_k_per_baseline[name], catalog=evaluation_catalog, k=args.k
                ),
            }

    # ---- Print table ----
    metric_cols = [f"NDCG@{args.k}", f"MAP@{args.k}",
                   f"Hit@{args.k}", f"Coverage@{args.k}"]
    header = f"{'baseline':<14}" + "".join(f"{m:>14}" for m in metric_cols)
    print(header)
    print("-" * len(header))
    for name in ["random", "popularity", "sports_v0", "ltr_v1"]:
        row = results[name]
        cells = "".join(f"{row[m]:>14.4f}" for m in metric_cols)
        print(f"{name:<14}{cells}")

    # ---- Feature importance (Task 20) ----
    try:
        store = ArtifactStore.load(args.artifacts_dir)
        imps = store.model.feature_importances_
        names = store.metadata["feature_names"]
        pairs = sorted(zip(names, imps), key=lambda p: -p[1])
        print("\nFeature importances (gain):")
        for name, imp in pairs:
            print(f"  {name:<22}{imp:>10}")
    except Exception as exc:
        print(f"\nFeature importance unavailable: {exc}")

    # ---- Per-segment Hit@k (Task 20) ----
    cold_queries = [q for q in queries if q["history_count"] == 0]
    power_queries = [q for q in queries if q["history_count"] >= 3]

    def segment_hit(baseline_name: str, segment: list[dict]) -> float:
        if not segment:
            return float("nan")
        scores = [
            hit_at_k(
                top_k_per_baseline[baseline_name][query["id"]],
                query["truth"],
                k=args.k,
            )
            for query in segment
        ]
        return float(np.mean(scores))

    print(f"\nPer-segment Hit@{args.k} (LTR):")
    print(f"  cold  (n={len(cold_queries):>3}): {segment_hit('ltr_v1', cold_queries):.4f}")
    print(f"  power (n={len(power_queries):>3}): {segment_hit('ltr_v1', power_queries):.4f}")

    # ---- Quality gate ----
    gate_ok = (
        results["ltr_v1"][f"NDCG@{args.k}"] > results["sports_v0"][f"NDCG@{args.k}"] + args.min_ndcg_vs_sports
        and results["ltr_v1"][f"Coverage@{args.k}"] >= args.min_coverage
    )
    print("\nGate:", "PASS" if gate_ok else "FAIL")

    # Save JSON next to artifacts for downstream tooling
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2)
    )

    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
