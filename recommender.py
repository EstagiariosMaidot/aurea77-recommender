"""LTR-based inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.artifacts import ArtifactStore, CachedArtifactStore
from ml.data_loader import (
    load_athlete_profiles,
    load_events,
    load_registrations,
    load_reviews,
)
from ml.embeddings import EMBEDDING_DIM, embed_sports_profile
from ml.features import EVENT_TRAINING_SCHEMA_VERSION, FEATURE_NAMES, build_feature_row

log = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class Recommendation:
    event_id: int
    score: float


def _chronological_fallback(
    conn: Any,
    *,
    user_id: int,
    limit: int,
    query_now: Optional[datetime],
    preloaded_data: Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> list[Recommendation]:
    """Return eligible events newest-first when a compatible model is absent."""
    if preloaded_data is None:
        events = load_events(conn)
        registrations = load_registrations(conn)
    else:
        events, registrations, _, _ = preloaded_data

    now = query_now if query_now is not None else datetime.now(timezone.utc).replace(tzinfo=None)
    user_past_ids = set(registrations.loc[
        registrations["user_id"] == user_id, "event_id"
    ].tolist())
    eligible = events[
        events["published_at"].notna()
        & (events["published_at"] <= pd.Timestamp(now))
        & (events["start_at"] > pd.Timestamp(now))
        & ~events["id"].isin(user_past_ids)
    ].sort_values(["published_at", "start_at"], ascending=False)
    return [Recommendation(event_id=int(event_id), score=0.0)
            for event_id in eligible["id"].head(limit)]


def recommend(
    conn: Any,
    user_id: int,
    limit: int = 10,
    *,
    artifacts_dir: Optional[Path] = None,
    query_now: Optional[datetime] = None,
    cached_store: Optional[CachedArtifactStore | ArtifactStore] = None,
    preloaded_data: Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
) -> list[Recommendation]:
    """Top-N recommendations.

    Args:
        query_now: Override for "current time" used to filter future candidates.
                   If None, uses datetime.now(UTC). Useful for offline evaluation.
        cached_store: Optional in-memory cache or already loaded artifact
                      store. When provided, artifacts_dir is ignored.
        preloaded_data: Optional ``(events, registrations, reviews, profiles)``
                        tuple for batch callers such as offline evaluation.
    """
    try:
        if cached_store is not None:
            store = (cached_store if isinstance(cached_store, ArtifactStore)
                     else cached_store.get())
        else:
            adir = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
            store = ArtifactStore.load(adir)
    except FileNotFoundError:
        log.warning("Artifacts missing — using chronological fallback")
        return _chronological_fallback(
            conn, user_id=user_id, limit=limit, query_now=query_now,
            preloaded_data=preloaded_data,
        )

    if (
        store.metadata.get("feature_names") != FEATURE_NAMES
        or store.metadata.get("training_schema_version") != EVENT_TRAINING_SCHEMA_VERSION
    ):
        log.warning(
            "Event artifacts use an incompatible feature schema; using chronological fallback"
        )
        return _chronological_fallback(
            conn, user_id=user_id, limit=limit, query_now=query_now,
            preloaded_data=preloaded_data,
        )

    return _recommend_with_model(
        conn,
        user_id=user_id,
        limit=limit,
        store=store,
        query_now=query_now,
        preloaded_data=preloaded_data,
    )


def _recommend_with_model(
    conn: Any,
    *,
    user_id: int,
    limit: int,
    store: ArtifactStore,
    query_now: Optional[datetime] = None,
    preloaded_data: Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
) -> list[Recommendation]:
    if preloaded_data is None:
        events = load_events(conn)
        registrations = load_registrations(conn)
        reviews = load_reviews(conn)
        profiles = load_athlete_profiles(conn)
    else:
        events, registrations, reviews, profiles = preloaded_data

    # When query_now is set (offline eval), only consider registrations before
    # that point — avoids leaking future ground-truth into the exclusion set.
    if query_now is not None:
        registrations = registrations[registrations["created_at"] < pd.Timestamp(query_now)]
        reviews = reviews[reviews["created_at"] < pd.Timestamp(query_now)]

    user_regs = registrations[registrations["user_id"] == user_id]
    user_past_ids = set(user_regs["event_id"].tolist())

    # Candidates: future, published, not already registered for
    now = query_now if query_now is not None else datetime.now(timezone.utc).replace(tzinfo=None)
    candidates = events[
        events["published_at"].notna()
        & (events["published_at"] <= pd.Timestamp(now))
        & (events["start_at"] > pd.Timestamp(now))
        & ~events["id"].isin(user_past_ids)
    ]
    if candidates.empty:
        return []

    # User aggregates
    user_profile = profiles[profiles["user_id"] == user_id]
    sports = tuple(sorted((user_profile.iloc[0]["sports_practiced"] if not user_profile.empty else []) or []))
    nationality = user_profile.iloc[0]["nationality"] if not user_profile.empty else None

    past = events[events["id"].isin(user_past_ids)][["id", "start_at", "latitude", "longitude"]]
    home_latlng = (past[["latitude", "longitude"]].dropna().mean().to_dict()
                   if not past.empty else {"latitude": None, "longitude": None})
    home_lat = home_latlng.get("latitude")
    home_lng = home_latlng.get("longitude")
    if pd.isna(home_lat): home_lat = None
    if pd.isna(home_lng): home_lng = None

    history_dates = past["start_at"].dropna().tolist()
    # User semantic profile
    rows = [store.event_id_to_row[eid] for eid in user_past_ids if eid in store.event_id_to_row]
    if rows:
        user_emb = store.event_embeddings[rows].mean(axis=0)
    else:
        user_emb = np.zeros(store.event_embeddings.shape[1] if store.event_embeddings.size else EMBEDDING_DIM,
                            dtype=np.float32)
    sports_emb = embed_sports_profile(sports)

    # Event-level aggregates
    popularity = registrations.groupby("event_id").size().to_dict()
    score_sum = reviews.groupby("event_id")["score"].sum().to_dict()
    score_cnt = reviews.groupby("event_id")["score"].count().to_dict()
    org_by_event = dict(zip(events["id"], events["organizer_id"]))
    org_score_sum, org_score_cnt = {}, {}
    for r in reviews.itertuples(index=False):
        org = org_by_event.get(r.event_id)
        if org is None:
            continue
        org_score_sum[org] = org_score_sum.get(org, 0.0) + r.score
        org_score_cnt[org] = org_score_cnt.get(org, 0) + 1

    # Feature matrix
    feat_rows = []
    candidate_ids = []
    for row in candidates.itertuples(index=False):
        eid = int(row.id)
        if eid not in store.event_id_to_row:
            continue
        emb = store.event_embeddings[store.event_id_to_row[eid]]
        sem_sim = float(np.dot(user_emb, emb))
        sports_similarity = float(np.dot(sports_emb, emb)) if sports else 0.0
        org_id = org_by_event.get(eid)
        feat_rows.append(build_feature_row(
            sem_sim=sem_sim,
            sports_similarity=sports_similarity,
            has_sports_practiced=bool(sports),
            user_home_lat=home_lat, user_home_lng=home_lng,
            user_nationality=nationality,
            user_history_dates=[pd.Timestamp(d).to_pydatetime() for d in history_dates],
            user_n_registrations=len(user_past_ids),
            event_lat=row.latitude, event_lng=row.longitude,
            event_country_code=row.country_code,
            event_start_at=pd.Timestamp(row.start_at).to_pydatetime(),
            query_now=now,
            event_popularity_raw=popularity.get(eid, 0),
            event_score_sum=score_sum.get(eid, 0.0),
            event_score_count=score_cnt.get(eid, 0),
            organizer_score_sum=org_score_sum.get(org_id, 0.0),
            organizer_score_count=org_score_cnt.get(org_id, 0),
        ))
        candidate_ids.append(eid)

    if not feat_rows:
        return []

    X = np.vstack(feat_rows)
    scores = store.model.predict(X)
    ranked = sorted(
        zip(candidate_ids, scores.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )[:limit]
    return [Recommendation(event_id=eid, score=float(s)) for eid, s in ranked]
