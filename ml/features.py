"""Pure feature-building functions for the LTR recommender."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np


EARTH_RADIUS_KM = 6371.0088
EVENT_TRAINING_SCHEMA_VERSION = 2


def haversine_km(
    lat1: Optional[float],
    lng1: Optional[float],
    lat2: Optional[float],
    lng2: Optional[float],
) -> float:
    """Great-circle distance in km. Returns NaN if any coord is None."""
    if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
        return float("nan")
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def jaccard_overlap(a: set[int], b: set[int]) -> float:
    """Jaccard similarity. Returns 0.0 when both sets are empty."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def bayesian_mean(
    sum_scores: float,
    n_votes: int,
    prior_mean: float = 3.0,
    prior_strength: int = 5,
) -> float:
    """Smoothed mean: shrinks toward `prior_mean` when votes are few."""
    return (sum_scores + prior_mean * prior_strength) / (n_votes + prior_strength)


def month_match_fraction(
    history: list[datetime], candidate_month: int
) -> float:
    """Fraction of `history` events whose month matches `candidate_month`."""
    if not history:
        return 0.0
    hits = sum(1 for d in history if d.month == candidate_month)
    return hits / len(history)


FEATURE_NAMES: list[str] = [
    "sem_sim",
    "sports_similarity",
    "has_sports_practiced",
    "same_country",
    "distance_km",
    "days_until_event",
    "month_match",
    "event_popularity",
    "event_avg_rating",
    "organizer_quality",
    "user_n_registrations",
]


def build_feature_row(
    *,
    sem_sim: float,
    sports_similarity: float,
    has_sports_practiced: bool,
    user_home_lat: float | None,
    user_home_lng: float | None,
    user_nationality: str | None,
    user_history_dates: list[datetime],
    user_n_registrations: int,
    event_lat: float | None,
    event_lng: float | None,
    event_country_code: str | None,
    event_start_at: datetime,
    query_now: datetime,
    event_popularity_raw: int,
    event_score_sum: float,
    event_score_count: int,
    organizer_score_sum: float,
    organizer_score_count: int,
) -> np.ndarray:
    """Produce an 11-element feature vector for one (user, candidate) pair."""

    same_country = 1.0 if (
        user_nationality and event_country_code
        and user_nationality.upper() == event_country_code.upper()
    ) else 0.0

    distance = haversine_km(user_home_lat, user_home_lng, event_lat, event_lng)
    days_until = (event_start_at - query_now).total_seconds() / 86400.0

    row = np.array([
        float(sem_sim),
        float(sports_similarity),
        float(has_sports_practiced),
        same_country,
        distance,
        days_until,
        month_match_fraction(user_history_dates, event_start_at.month),
        float(np.log1p(event_popularity_raw)),
        bayesian_mean(event_score_sum, event_score_count),
        bayesian_mean(organizer_score_sum, organizer_score_count),
        float(user_n_registrations),
    ], dtype=np.float64)
    return row
