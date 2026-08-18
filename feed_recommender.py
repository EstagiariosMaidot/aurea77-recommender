"""Serve top-N mixed-feed recommendations for a viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml.feed_candidate_pools import (
    _friend_ids,
    build_discovery_pool,
    build_friend_pool,
)
from ml.feed_data_loader import (
    load_feed_posts,
    load_friendships,
    load_interactions_log,
    load_post_impressions,
    load_post_media,
    load_user_blocks,
    load_user_mutes,
    load_users_min,
)
from ml.feed_diversity import cap_per_author, enforce_source_floor
from ml.feed_features import build_feature_row
from ml.feed_labels import POSITIVE_INTERACTION_TYPES
from ml.feed_utils import open_connection, viewer_profile_embedding


load_dotenv()


@dataclass(frozen=True)
class FeedRecommendation:
    post_id: int
    source: str
    score: float


def feed_model_status(artifacts_dir: str = "artifacts/feed") -> tuple[str, str | None]:
    """Return the serving mode and model version for the feed artifacts."""
    root = Path(artifacts_dir)
    required_artifacts = (
        root / "model.pkl",
        root / "post_id_to_row.json",
        root / "post_embeddings.npy",
    )
    if not all(path.exists() for path in required_artifacts):
        return "chronological_fallback", None

    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return "recommender", None

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "recommender", None

    # ``train_feed`` deliberately records zero positives when the telemetry
    # tables have not accumulated enough feedback.  Those artifacts are useful
    # for local fixtures, but are not a trained ranker and must never make the
    # serving endpoint advertise personalized ranking.
    if int(metadata.get("n_positives", 0)) <= 0:
        return "chronological_fallback", None

    version = metadata.get("model_version")
    return "recommender", str(version) if version else None


def recommend_for_viewer(
    *,
    viewer_id: int,
    db_path: str | None = None,
    artifacts_dir: str = "artifacts/feed",
    limit: int = 10,
    discovery_floor: float = 0.15,
    cap_per_author_n: int = 2,
) -> list[FeedRecommendation]:
    """Return top-N recommendations for a viewer, with source labels.

    If feed artifacts have not been trained yet, return the same eligible
    candidates in reverse chronological order. This keeps automatic feed
    serving available while preserving the existing privacy, friendship, block
    and mute constraints.
    """
    root = Path(artifacts_dir)

    conn = open_connection(db_path)
    try:
        posts = load_feed_posts(conn)
        users = load_users_min(conn)
        friendships = load_friendships(conn)
        media = load_post_media(conn)
        blocks = load_user_blocks(conn)
        mutes = load_user_mutes(conn)
        interactions = load_interactions_log(conn)
        impressions = load_post_impressions(conn)
    finally:
        conn.close()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    friend_pool = build_friend_pool(
        viewer_id=viewer_id, posts=posts, friendships=friendships,
        blocks=blocks, mutes=mutes, now=now, window_days=14,
    )
    discovery_pool = build_discovery_pool(
        viewer_id=viewer_id, posts=posts, friendships=friendships,
        blocks=blocks, mutes=mutes, now=now, window_days=7,
        min_engagement=0,
    )

    candidates = list(dict.fromkeys(friend_pool + discovery_pool))
    if not candidates:
        return []

    sources = {pid: ("friend" if pid in set(friend_pool) else "discovery") for pid in candidates}
    authors = {int(row["id"]): int(row["user_id"]) for _, row in posts.iterrows()}

    ranking_mode, _ = feed_model_status(str(root))
    if ranking_mode == "chronological_fallback":
        ranked = (
            posts[posts["id"].isin(candidates)]
            .sort_values("created_at", ascending=False)["id"]
            .astype(int)
            .tolist()
        )
        top = ranked[:limit]
        reserves = [pid for pid in ranked[limit:] if sources[pid] == "discovery"]
        top = enforce_source_floor(
            ranked_ids=top,
            reserves=reserves,
            sources=sources,
            source="discovery",
            floor_ratio=discovery_floor,
        )
        top = cap_per_author(top, authors=authors, cap=cap_per_author_n)
        return [FeedRecommendation(post_id=pid, source=sources[pid], score=0.0) for pid in top[:limit]]

    model = joblib.load(root / "model.pkl")
    id_to_row = {int(k): int(v) for k, v in json.loads((root / "post_id_to_row.json").read_text()).items()}
    matrix = np.load(root / "post_embeddings.npy")

    fi = _friend_ids(viewer_id, friendships)
    viewer_positives = list(interactions[
        (interactions["user_id"] == viewer_id)
        & (interactions["interaction_type"].isin(POSITIVE_INTERACTION_TYPES))
    ]["feed_post_id"].astype(int).tolist())
    profile = viewer_profile_embedding(matrix, id_to_row, viewer_positives)

    rows: list[dict[str, float]] = []
    scored_ids: list[int] = []
    for pid in candidates:
        if pid not in id_to_row:
            continue
        author_id = int(posts[posts["id"] == pid].iloc[0]["user_id"])
        rows.append(build_feature_row(
            viewer_id=viewer_id, post_id=pid, author_id=author_id,
            posts=posts, users=users, media=media,
            friendships=friendships, friend_ids=fi,
            interactions=interactions, impressions=impressions,
            viewer_profile_embedding=profile,
            post_embedding=matrix[id_to_row[pid]],
            now=now,
        ))
        scored_ids.append(pid)

    if model is None or not rows:
        ranked = (
            posts[posts["id"].isin(scored_ids)]
            .sort_values("created_at", ascending=False)["id"]
            .astype(int)
            .tolist()
        )
        scores_by_id: dict[int, float] = {pid: 0.0 for pid in scored_ids}
    else:
        X = pd.DataFrame(rows)
        scores = model.predict(X)
        order = np.argsort(-scores)
        ranked = [scored_ids[i] for i in order]
        scores_by_id = {scored_ids[i]: float(scores[i]) for i in range(len(scored_ids))}

    # Guardrails
    top = ranked[:limit]
    reserves = [pid for pid in ranked[limit:] if sources[pid] == "discovery"]
    top = enforce_source_floor(
        ranked_ids=top,
        reserves=reserves,
        sources=sources,
        source="discovery",
        floor_ratio=discovery_floor,
    )
    top = cap_per_author(top, authors=authors, cap=cap_per_author_n)

    return [FeedRecommendation(post_id=pid, source=sources[pid], score=scores_by_id.get(pid, 0.0)) for pid in top[:limit]]
