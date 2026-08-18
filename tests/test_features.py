# tests/test_features.py
import math
from datetime import datetime

import numpy as np

from ml.features import haversine_km, jaccard_overlap, bayesian_mean, month_match_fraction
from ml.features import FEATURE_NAMES, build_feature_row


def test_haversine_zero_distance():
    assert haversine_km(38.7, -9.1, 38.7, -9.1) == 0.0


def test_haversine_lisboa_porto_approx_275km():
    # Lisboa ↔ Porto: ~275 km ground-truth
    d = haversine_km(38.7, -9.1, 41.15, -8.6)
    assert 265 < d < 285


def test_haversine_returns_nan_when_any_coord_is_none():
    assert math.isnan(haversine_km(None, -9.1, 41.15, -8.6))
    assert math.isnan(haversine_km(38.7, None, 41.15, -8.6))
    assert math.isnan(haversine_km(38.7, -9.1, None, -8.6))
    assert math.isnan(haversine_km(38.7, -9.1, 41.15, None))


def test_jaccard_identical_sets():
    assert jaccard_overlap({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint():
    assert jaccard_overlap({1, 2}, {3, 4}) == 0.0


def test_jaccard_partial():
    assert abs(jaccard_overlap({1, 2}, {2, 3}) - 1 / 3) < 1e-9


def test_jaccard_empty_inputs_return_zero():
    assert jaccard_overlap(set(), set()) == 0.0
    assert jaccard_overlap({1}, set()) == 0.0
    assert jaccard_overlap(set(), {1}) == 0.0


def test_bayesian_mean_zero_votes_returns_prior():
    assert bayesian_mean(sum_scores=0, n_votes=0, prior_mean=3.0, prior_strength=5) == 3.0


def test_bayesian_mean_many_votes_approaches_observed_mean():
    result = bayesian_mean(sum_scores=4500, n_votes=1000, prior_mean=3.0, prior_strength=5)
    assert abs(result - 4.4925) < 0.01


def test_bayesian_mean_few_votes_pulled_toward_prior():
    result = bayesian_mean(sum_scores=5, n_votes=1, prior_mean=3.0, prior_strength=5)
    assert abs(result - (20 / 6)) < 1e-9


def test_month_match_no_history_returns_zero():
    assert month_match_fraction([], candidate_month=5) == 0.0


def test_month_match_all_same_month_returns_one():
    history = [
        datetime(2025, 5, 1),
        datetime(2024, 5, 15),
        datetime(2023, 5, 30),
    ]
    assert month_match_fraction(history, candidate_month=5) == 1.0


def test_month_match_partial():
    history = [datetime(2025, 5, 1), datetime(2025, 6, 1), datetime(2025, 7, 1)]
    assert month_match_fraction(history, candidate_month=5) == 1 / 3


def test_feature_row_has_expected_length_and_names():
    assert len(FEATURE_NAMES) == 11
    expected = [
        "sem_sim", "sports_similarity", "has_sports_practiced",
        "same_country", "distance_km",
        "days_until_event", "month_match",
        "event_popularity", "event_avg_rating", "organizer_quality",
        "user_n_registrations",
    ]
    assert FEATURE_NAMES == expected


def test_feature_row_warm_user():
    row = build_feature_row(
        sem_sim=0.8,
        sports_similarity=0.75,
        has_sports_practiced=True,
        user_home_lat=38.7, user_home_lng=-9.1,
        user_nationality="PT",
        user_history_dates=[datetime(2025, 5, 1), datetime(2025, 7, 1)],
        user_n_registrations=2,
        event_lat=38.7, event_lng=-9.1,
        event_country_code="PT",
        event_start_at=datetime(2026, 5, 10),
        query_now=datetime(2026, 4, 16),
        event_popularity_raw=7,
        event_score_sum=18, event_score_count=4,
        organizer_score_sum=50, organizer_score_count=12,
    )
    assert row.shape == (11,)
    assert row[FEATURE_NAMES.index("sem_sim")] == 0.8
    assert row[FEATURE_NAMES.index("sports_similarity")] == 0.75
    assert row[FEATURE_NAMES.index("has_sports_practiced")] == 1.0
    assert row[FEATURE_NAMES.index("same_country")] == 1.0
    assert row[FEATURE_NAMES.index("distance_km")] == 0.0
    assert row[FEATURE_NAMES.index("days_until_event")] == 24.0
    assert row[FEATURE_NAMES.index("month_match")] == 0.5
    assert abs(row[FEATURE_NAMES.index("event_popularity")] - np.log1p(7)) < 1e-9
    assert abs(row[FEATURE_NAMES.index("event_avg_rating")] - 33 / 9) < 1e-9
    assert row[FEATURE_NAMES.index("user_n_registrations")] == 2.0


def test_feature_row_cold_user_produces_nans_where_expected():
    row = build_feature_row(
        sem_sim=0.0,
        sports_similarity=0.0, has_sports_practiced=False,
        user_home_lat=None, user_home_lng=None,
        user_nationality=None, user_history_dates=[],
        user_n_registrations=0,
        event_lat=38.7, event_lng=-9.1,
        event_country_code="PT",
        event_start_at=datetime(2026, 5, 10),
        query_now=datetime(2026, 4, 16),
        event_popularity_raw=7,
        event_score_sum=18, event_score_count=4,
        organizer_score_sum=0, organizer_score_count=0,
    )
    assert row[FEATURE_NAMES.index("same_country")] == 0.0
    assert np.isnan(row[FEATURE_NAMES.index("distance_km")])
    assert row[FEATURE_NAMES.index("user_n_registrations")] == 0.0
    assert row[FEATURE_NAMES.index("organizer_quality")] == 3.0
