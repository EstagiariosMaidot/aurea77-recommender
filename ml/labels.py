"""Graded-relevance labels + negative sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def graded_relevance(
    registrations: pd.DataFrame, reviews: pd.DataFrame
) -> dict[tuple[int, int], int]:
    """Return {(user_id, event_id): relevance} in {1, 2, 3}.

    - registered/completed + score >= 4 → 3
    - registered/completed (no high review)     → 2
    - planned                                    → 1
    """
    rel: dict[tuple[int, int], int] = {}
    for row in registrations.itertuples(index=False):
        key = (row.user_id, row.event_id)
        if row.status == "planned":
            rel[key] = 1
        else:
            rel[key] = 2
    for row in reviews.itertuples(index=False):
        key = (row.user_id, row.event_id)
        if row.score >= 4 and key in rel:
            rel[key] = 3
    return rel


def sample_negatives(
    *,
    registrations: pd.DataFrame,
    events: pd.DataFrame,
    n_per_positive: int,
    hard_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """For each positive (user, event, created_at), sample N negatives.

    Constraints:
    - Negative events must be published and start after `created_at`.
    - Negative events must not be the positive itself or any other event the
      user registered for.
    - `hard_fraction` of the negatives come from the same-country hard pool;
      the rest are sampled uniformly. Categories are intentionally not used.
    """
    event_pub = dict(zip(events["id"], events["published_at"]))
    event_start = dict(zip(events["id"], events["start_at"]))
    event_country = dict(zip(events["id"], events["country_code"]))

    out_rows: list[dict] = []

    for row in registrations.itertuples(index=False):
        uid = row.user_id
        pos_event_id = row.event_id
        reg_at = row.created_at

        pos_country = event_country.get(pos_event_id)

        # The user's later registrations must not affect a query in the past.
        # Same-timestamp registrations are treated as one recommendation
        # request and are excluded together.
        forbidden = set(registrations.loc[
            (registrations["user_id"] == uid)
            & (registrations["created_at"] <= reg_at),
            "event_id",
        ].tolist())
        pool: list[int] = [
            eid for eid, pub in event_pub.items()
            if (
                eid not in forbidden
                and pd.notna(pub) and pub < reg_at
                and pd.notna(event_start.get(eid)) and event_start[eid] > reg_at
            )
        ]
        if not pool:
            continue

        hard_pool = [
            eid for eid in pool
            if event_country.get(eid) == pos_country
        ]
        n_hard = min(int(round(n_per_positive * hard_fraction)), len(hard_pool))
        n_random = n_per_positive - n_hard

        sampled: list[int] = []
        if n_hard:
            sampled += list(rng.choice(hard_pool, size=n_hard, replace=False))
        if n_random:
            remaining = [e for e in pool if e not in sampled]
            n_random = min(n_random, len(remaining))
            if n_random > 0:
                sampled += list(rng.choice(remaining, size=n_random, replace=False))

        for eid in sampled:
            out_rows.append({
                "user_id": uid,
                "event_id": int(eid),
                "relevance": 0,
                "created_at": reg_at,
            })

    return pd.DataFrame(
        out_rows, columns=["user_id", "event_id", "relevance", "created_at"]
    )
