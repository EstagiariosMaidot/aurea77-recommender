"""Label building and negative sampling for the mixed-feed ranker.

Positive pairs come from ``feed_post_interactions_log`` when the interaction
type expresses a positive signal. Negative pairs come from
``post_impressions`` — impressions the viewer saw but didn't convert.

The sampler stratifies by ``source`` so the training set has a controlled
balance of friend vs discovery negatives even when production distribution
skews heavily to one side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


POSITIVE_INTERACTION_TYPES: set[str] = {
    "like",
    "comment",
    "share",
    "follow_from_post",
}


def build_positive_pairs(interactions: pd.DataFrame) -> set[tuple[int, int]]:
    """Return the set of (user_id, feed_post_id) that count as positives."""
    if interactions.empty:
        return set()
    mask = interactions["interaction_type"].isin(POSITIVE_INTERACTION_TYPES)
    positives = interactions[mask][["user_id", "feed_post_id"]]
    return {
        (int(uid), int(pid))
        for uid, pid in zip(positives["user_id"], positives["feed_post_id"])
    }


def build_negative_pairs(
    impressions: pd.DataFrame,
    positives: set[tuple[int, int]],
) -> pd.DataFrame:
    """Return impressions that did NOT result in a positive interaction.

    The returned DataFrame preserves the ``source`` column so the downstream
    sampler can enforce a stratified balance.
    """
    if impressions.empty:
        return impressions.iloc[0:0].copy()

    keys = list(zip(impressions["user_id"].astype(int), impressions["feed_post_id"].astype(int)))
    mask = np.array([k not in positives for k in keys])
    return impressions[mask].reset_index(drop=True)


def sample_negatives_stratified_by_source(
    negatives: pd.DataFrame,
    *,
    friend_fraction: float,
    total: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw ``total`` negatives with the target friend/discovery split.

    Parameters
    ----------
    negatives:
        Pool of candidate negatives (typically the output of
        :func:`build_negative_pairs`) that includes a ``source`` column.
    friend_fraction:
        Desired share of friend-source negatives in the sample. Must be in [0, 1].
    total:
        Total number of negatives to sample. When the pool is smaller than the
        target for a given source, every available row is used (no oversampling).
    rng:
        Deterministic numpy generator so training runs are reproducible.
    """
    if not 0.0 <= friend_fraction <= 1.0:
        raise ValueError("friend_fraction must be in [0, 1]")
    if negatives.empty or total <= 0:
        return negatives.iloc[0:0].copy()

    target_friend = int(round(total * friend_fraction))
    target_discovery = total - target_friend

    parts: list[pd.DataFrame] = []
    for source_key, target in (("friend", target_friend), ("discovery", target_discovery)):
        pool = negatives[negatives["source"] == source_key]
        take = min(len(pool), target)
        if take > 0:
            picked_idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            parts.append(pool.loc[picked_idx])

    if not parts:
        return negatives.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)
