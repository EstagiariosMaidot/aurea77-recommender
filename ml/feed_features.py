"""Pure feature functions for the mixed-feed ranker.

Each function takes explicit inputs (no hidden state) and returns a scalar.
This design makes them trivially testable and easy to combine into a feature
matrix later (see Task 10).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------- Content features ----------


def feature_has_media(*, post_id: int, media: pd.DataFrame) -> int:
    """Return 1 if the post has at least one media row, else 0."""
    if media.empty:
        return 0
    return int((media["post_id"] == post_id).any())


_MEDIA_TYPE_CODE = {"image": 1, "video": 2}


def feature_media_type_code(*, post_id: int, media: pd.DataFrame) -> int:
    """0 = no media, 1 = image, 2 = video (based on the first media row)."""
    if media.empty:
        return 0
    rows = media[media["post_id"] == post_id]
    if rows.empty:
        return 0
    return _MEDIA_TYPE_CODE.get(str(rows.iloc[0]["media_type"]).lower(), 0)


def feature_body_length_bucket(body: str | None) -> int:
    """0 = short (<50 chars), 1 = medium, 2 = long (>=300 chars)."""
    if body is None:
        return 0
    n = len(str(body).strip())
    if n < 50:
        return 0
    if n < 300:
        return 1
    return 2


# ---------- Temporal features ----------


def feature_post_age_hours(*, created_at: datetime, now: datetime) -> float:
    """Hours since the post was created. Clamped to zero for future timestamps."""
    delta = (now - created_at).total_seconds() / 3600.0
    return max(0.0, delta)


# ---------- Engagement features ----------


def _days_since(created_at: datetime, now: datetime) -> float:
    days = (now - created_at).total_seconds() / 86400.0
    # Avoid inflating counts for very fresh posts.
    return max(1.0, days)


def feature_likes_normalized(*, likes_count: int, created_at: datetime, now: datetime) -> float:
    """Likes per day since post creation (min 1 day)."""
    return float(likes_count) / _days_since(created_at, now)


def feature_comments_normalized(*, comments_count: int, created_at: datetime, now: datetime) -> float:
    """Comments per day since post creation (min 1 day)."""
    return float(comments_count) / _days_since(created_at, now)


def feature_engagement_velocity(
    *,
    post_id: int,
    interactions: pd.DataFrame,
    now: datetime,
    window_hours: int = 6,
) -> int:
    """Number of interactions on this post in the last ``window_hours``."""
    if interactions.empty:
        return 0
    cutoff = now - timedelta(hours=window_hours)
    mask = (
        (interactions["feed_post_id"] == post_id)
        & (interactions["occurred_at"] >= cutoff)
    )
    return int(mask.sum())


# ---------- Social / author features ----------


def feature_is_friend(*, author_id: int, friend_ids: set[int]) -> int:
    """1 if the author is a friend of the viewer, else 0."""
    return int(author_id in friend_ids)


def feature_friendship_age_days(
    *,
    viewer_id: int,
    author_id: int,
    friendships: pd.DataFrame,
    now: datetime,
) -> float:
    """Days since the accepted friendship was established.

    Returns NaN when viewer and author are not friends. Friendships are stored
    as (requester_id, receiver_id) pairs — order doesn't matter.
    """
    if friendships.empty:
        return float("nan")
    mask = (
        ((friendships["requester_id"] == viewer_id) & (friendships["receiver_id"] == author_id))
        | ((friendships["requester_id"] == author_id) & (friendships["receiver_id"] == viewer_id))
    )
    matches = friendships[mask]
    if matches.empty:
        return float("nan")
    established = pd.to_datetime(matches.iloc[0]["created_at"])
    return (now - established).total_seconds() / 86400.0


def feature_author_interaction_rate(
    *,
    viewer_id: int,
    author_id: int,
    interactions: pd.DataFrame,
    impressions: pd.DataFrame,
    posts: pd.DataFrame,
) -> float:
    """Interactions viewer→author / impressions of author's posts served to viewer.

    Returns 0.0 when the denominator is zero.
    """
    if posts.empty:
        return 0.0
    author_post_ids = set(posts[posts["user_id"] == author_id]["id"].astype(int).tolist())
    if not author_post_ids:
        return 0.0

    if impressions.empty:
        return 0.0
    impressions_by_viewer_on_author_posts = impressions[
        (impressions["user_id"] == viewer_id)
        & (impressions["feed_post_id"].isin(author_post_ids))
    ]
    denominator = len(impressions_by_viewer_on_author_posts)
    if denominator == 0:
        return 0.0

    if interactions.empty:
        return 0.0
    interactions_by_viewer_on_author_posts = interactions[
        (interactions["user_id"] == viewer_id)
        & (interactions["feed_post_id"].isin(author_post_ids))
    ]
    numerator = len(interactions_by_viewer_on_author_posts)
    return float(numerator) / float(denominator)


def feature_affinity_via_friends(
    *,
    author_id: int,
    friend_ids: set[int],
    interactions: pd.DataFrame,
    posts: pd.DataFrame,
) -> float:
    """Fraction of viewer's friends that interacted with any post of the author.

    Returns 0.0 when viewer has no friends. Range: [0, 1].
    """
    if not friend_ids:
        return 0.0

    if posts.empty or interactions.empty:
        return 0.0

    author_post_ids = set(posts[posts["user_id"] == author_id]["id"].astype(int).tolist())
    if not author_post_ids:
        return 0.0

    friends_who_interacted = set(
        interactions[
            (interactions["user_id"].isin(friend_ids))
            & (interactions["feed_post_id"].isin(author_post_ids))
        ]["user_id"].astype(int).tolist()
    )
    return len(friends_who_interacted) / len(friend_ids)


# ---------- Cold-start / viewer features ----------


def feature_viewer_n_interactions(
    *,
    viewer_id: int,
    interactions: pd.DataFrame,
    now: datetime,
    days: int = 90,
) -> int:
    """Total interactions by the viewer within the last ``days`` days."""
    if interactions.empty:
        return 0
    cutoff = now - timedelta(days=days)
    mask = (
        (interactions["user_id"] == viewer_id)
        & (interactions["occurred_at"] >= cutoff)
    )
    return int(mask.sum())


def feature_viewer_n_friends(friend_ids: set[int]) -> int:
    """Number of accepted friends the viewer has."""
    return len(friend_ids)


def feature_viewer_days_since_signup(
    *,
    viewer_id: int,
    users: pd.DataFrame,
    now: datetime,
) -> float:
    """Days since the viewer registered. NaN when the user is unknown."""
    if users.empty:
        return float("nan")
    matches = users[users["id"] == viewer_id]
    if matches.empty:
        return float("nan")
    signup = pd.to_datetime(matches.iloc[0]["created_at"])
    return (now - signup).total_seconds() / 86400.0


# ---------- Semantic feature ----------


def feature_sem_sim(post_embedding: np.ndarray, viewer_profile_embedding: np.ndarray) -> float:
    """Cosine similarity between post embedding and viewer's profile embedding.

    Returns 0.0 if either vector has zero norm. The embeddings are usually
    already L2-normalised, so this reduces to a dot product in practice.
    """
    a_norm = np.linalg.norm(post_embedding)
    b_norm = np.linalg.norm(viewer_profile_embedding)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(post_embedding, viewer_profile_embedding) / (a_norm * b_norm))


# ---------- Assembly ----------


def build_feature_row(
    *,
    viewer_id: int,
    post_id: int,
    author_id: int,
    posts: pd.DataFrame,
    users: pd.DataFrame,
    media: pd.DataFrame,
    friendships: pd.DataFrame,
    friend_ids: set[int],
    interactions: pd.DataFrame,
    impressions: pd.DataFrame,
    viewer_profile_embedding: np.ndarray,
    post_embedding: np.ndarray,
    now: datetime,
) -> dict[str, int | float]:
    """Assemble the full feature dictionary for a (viewer, post) candidate pair.

    The output keys mirror the model input signature and are stable across
    training and inference.
    """
    post_row = posts[posts["id"] == post_id]
    if post_row.empty:
        raise ValueError(f"Post id {post_id} not found in posts DataFrame.")
    post = post_row.iloc[0]

    return {
        # Content
        "has_media": feature_has_media(post_id=post_id, media=media),
        "media_type_code": feature_media_type_code(post_id=post_id, media=media),
        "body_length_bucket": feature_body_length_bucket(post.get("body")),
        # Temporal
        "post_age_hours": feature_post_age_hours(created_at=post["created_at"], now=now),
        # Engagement
        "likes_normalized": feature_likes_normalized(
            likes_count=int(post["likes_count"]),
            created_at=post["created_at"],
            now=now,
        ),
        "comments_normalized": feature_comments_normalized(
            comments_count=int(post["comments_count"]),
            created_at=post["created_at"],
            now=now,
        ),
        "engagement_velocity": feature_engagement_velocity(
            post_id=post_id, interactions=interactions, now=now,
        ),
        # Social
        "is_friend": feature_is_friend(author_id=author_id, friend_ids=friend_ids),
        "friendship_age_days": feature_friendship_age_days(
            viewer_id=viewer_id, author_id=author_id, friendships=friendships, now=now,
        ),
        "author_interaction_rate": feature_author_interaction_rate(
            viewer_id=viewer_id, author_id=author_id,
            interactions=interactions, impressions=impressions, posts=posts,
        ),
        "affinity_via_friends": feature_affinity_via_friends(
            author_id=author_id,
            friend_ids=friend_ids, interactions=interactions, posts=posts,
        ),
        # Cold-start
        "viewer_n_interactions": feature_viewer_n_interactions(
            viewer_id=viewer_id, interactions=interactions, now=now,
        ),
        "viewer_n_friends": feature_viewer_n_friends(friend_ids),
        "viewer_days_since_signup": feature_viewer_days_since_signup(
            viewer_id=viewer_id, users=users, now=now,
        ),
        # Semantic
        "sem_sim": feature_sem_sim(post_embedding, viewer_profile_embedding),
    }
