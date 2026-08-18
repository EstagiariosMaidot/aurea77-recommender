"""Build the candidate pools that feed the mixed-feed ranker.

Two independent pools:
  * ``build_friend_pool`` — recent posts from users the viewer follows,
    filtered by privacy, blocks and mutes.
  * ``build_discovery_pool`` — recent public posts from users the viewer does
    NOT follow, pre-filtered by popularity, quality and recency (§6.2 of the
    Plan D spec).

Both return a list of ``feed_post_id`` integers ready for scoring.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def _friend_ids(viewer_id: int, friendships: pd.DataFrame) -> set[int]:
    if friendships.empty:
        return set()
    as_requester = friendships[friendships["requester_id"] == viewer_id]["receiver_id"]
    as_receiver = friendships[friendships["receiver_id"] == viewer_id]["requester_id"]
    return set(as_requester.tolist()) | set(as_receiver.tolist())


def _blocked_by_viewer(viewer_id: int, blocks: pd.DataFrame) -> set[int]:
    if blocks.empty:
        return set()
    return set(blocks[blocks["blocker_id"] == viewer_id]["blocked_id"].tolist())


def _muted_by_viewer(viewer_id: int, mutes: pd.DataFrame) -> set[int]:
    if mutes.empty:
        return set()
    return set(mutes[mutes["muter_id"] == viewer_id]["muted_id"].tolist())


def build_friend_pool(
    *,
    viewer_id: int,
    posts: pd.DataFrame,
    friendships: pd.DataFrame,
    blocks: pd.DataFrame,
    mutes: pd.DataFrame,
    now: datetime,
    window_days: int = 14,
) -> list[int]:
    """Return post IDs from the viewer's friends within the recency window."""
    friends = _friend_ids(viewer_id, friendships)
    if not friends:
        return []

    blocked = _blocked_by_viewer(viewer_id, blocks)
    muted = _muted_by_viewer(viewer_id, mutes)
    excluded_authors = (blocked | muted | {viewer_id})

    eligible_authors = friends - excluded_authors
    if not eligible_authors:
        return []

    cutoff = now - timedelta(days=window_days)
    candidates = posts[
        (posts["user_id"].isin(eligible_authors))
        & (posts["created_at"] >= cutoff)
        & (posts["privacy"].isin({"public", "friends", "members"}))
    ]
    return candidates["id"].astype(int).tolist()


def build_discovery_pool(
    *,
    viewer_id: int,
    posts: pd.DataFrame,
    friendships: pd.DataFrame,
    blocks: pd.DataFrame,
    mutes: pd.DataFrame,
    now: datetime,
    window_days: int = 7,
    min_engagement: int = 1,
) -> list[int]:
    """Return public post IDs from users the viewer does NOT follow.

    Pre-filtered by recency (``window_days``) and by minimal engagement
    (``likes_count + comments_count >= min_engagement``). Blocks and mutes
    are honoured on the author.
    """
    friends = _friend_ids(viewer_id, friendships)
    blocked = _blocked_by_viewer(viewer_id, blocks)
    muted = _muted_by_viewer(viewer_id, mutes)
    excluded_authors = (friends | blocked | muted | {viewer_id})

    cutoff = now - timedelta(days=window_days)
    engagement = posts["likes_count"] + posts["comments_count"]

    candidates = posts[
        (~posts["user_id"].isin(excluded_authors))
        & (posts["created_at"] >= cutoff)
        & (posts["privacy"] == "public")
        & (engagement >= min_engagement)
    ]
    return candidates["id"].astype(int).tolist()
