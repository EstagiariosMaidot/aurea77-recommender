"""Baseline rankers used as reference points in feed evaluation."""

from __future__ import annotations

import pandas as pd


def rank_chronological(*, candidates: list[int], posts: pd.DataFrame) -> list[int]:
    """Return candidates sorted from newest to oldest by ``created_at``."""
    if not candidates:
        return []
    subset = posts[posts["id"].isin(candidates)]
    ordered = subset.sort_values("created_at", ascending=False)
    return ordered["id"].astype(int).tolist()


def rank_popularity(*, candidates: list[int], posts: pd.DataFrame) -> list[int]:
    """Return candidates sorted by ``likes_count + comments_count`` desc."""
    if not candidates:
        return []
    subset = posts[posts["id"].isin(candidates)].copy()
    subset["_engagement"] = subset["likes_count"] + subset["comments_count"]
    ordered = subset.sort_values(["_engagement", "created_at"], ascending=[False, False])
    return ordered["id"].astype(int).tolist()


def rank_friends_only(
    *,
    candidates: list[int],
    posts: pd.DataFrame,
    friend_ids: set[int],
) -> list[int]:
    """Chronological ranking restricted to authors in ``friend_ids``."""
    if not candidates:
        return []
    subset = posts[
        posts["id"].isin(candidates)
        & posts["user_id"].isin(friend_ids)
    ]
    ordered = subset.sort_values("created_at", ascending=False)
    return ordered["id"].astype(int).tolist()
