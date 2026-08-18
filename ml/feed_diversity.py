"""Post-ranking guardrails for the mixed feed.

Two independent guarantees, applied in order:
1. ``cap_per_author``   — no author dominates the top of the feed.
2. ``enforce_source_floor`` — at least ``floor_ratio`` of the top slots come
   from a given source (e.g. discovery), backfilling from a reserve pool if
   the model didn't rank enough of them by itself.
"""

from __future__ import annotations

from typing import Mapping


def cap_per_author(
    ranked_ids: list[int],
    *,
    authors: Mapping[int, int],
    cap: int = 2,
) -> list[int]:
    """Return ``ranked_ids`` with at most ``cap`` posts per author.

    Preserves the original ranking order for the kept posts. Posts whose author
    already appeared ``cap`` times higher in the list are dropped.
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")

    counts: dict[int, int] = {}
    kept: list[int] = []
    for pid in ranked_ids:
        author = authors[pid]
        if counts.get(author, 0) < cap:
            kept.append(pid)
            counts[author] = counts.get(author, 0) + 1
    return kept


def enforce_source_floor(
    *,
    ranked_ids: list[int],
    reserves: list[int],
    sources: Mapping[int, str],
    source: str,
    floor_ratio: float,
) -> list[int]:
    """Ensure at least ``floor_ratio`` of ``ranked_ids`` come from ``source``.

    Parameters
    ----------
    ranked_ids:
        Original ranking (top-N).
    reserves:
        Extra IDs from the target ``source`` that were not selected by the
        model but can be promoted to fill the floor. Preserves reserve order.
    sources:
        Full mapping of ``id → source label``.
    source:
        The source that has a minimum floor (e.g. ``"discovery"``).
    floor_ratio:
        Value in [0, 1]. Minimum share of ``source`` slots in the result.

    Returns
    -------
    A list with the same length as ``ranked_ids``. If the model already
    satisfies the floor, the input is returned unchanged. Otherwise, the
    lowest-ranked non-``source`` items are demoted and replaced by reserve
    items from ``source``.
    """
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError("floor_ratio must be in [0, 1]")

    n = len(ranked_ids)
    if n == 0:
        return []

    needed = int(round(n * floor_ratio))
    have = sum(1 for pid in ranked_ids if sources.get(pid) == source)
    deficit = needed - have
    if deficit <= 0:
        return list(ranked_ids)

    # Take from reserves in order, respecting the target source.
    fill = [pid for pid in reserves if sources.get(pid) == source][:deficit]
    if not fill:
        return list(ranked_ids)

    # Drop the lowest-ranked non-source items to make room, preserving order otherwise.
    result = list(ranked_ids)
    removed = 0
    for i in range(len(result) - 1, -1, -1):
        if removed == len(fill):
            break
        if sources.get(result[i]) != source:
            result.pop(i)
            removed += 1

    return result + fill
