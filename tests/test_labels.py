# tests/test_labels.py
import numpy as np
import pandas as pd

from ml.labels import graded_relevance, sample_negatives


def test_graded_relevance_maps_high_review_to_3():
    regs = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "event_id": [10, 20, 30, 40],
        "status": ["completed", "completed", "planned", "completed"],
    })
    reviews = pd.DataFrame({
        "user_id": [1, 1],
        "event_id": [10, 20],
        "score": [5, 3],
    })
    rel = graded_relevance(regs, reviews)
    assert rel[(1, 10)] == 3
    assert rel[(1, 20)] == 2
    assert rel[(1, 30)] == 1
    assert rel[(2, 40)] == 2


def test_sample_negatives_respects_temporal_validity():
    registrations = pd.DataFrame({
        "user_id": [1], "event_id": [10],
        "created_at": pd.to_datetime(["2025-06-01"]),
    })
    events = pd.DataFrame({
        "id":           [10, 20, 30, 40, 50],
        "country_code": ["PT", "PT", "PT", "ES", "PT"],
        "published_at": pd.to_datetime([
            "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-07-01",
        ]),
        "start_at": pd.to_datetime([
            "2025-06-10", "2025-06-10", "2025-06-10", "2025-06-10", "2025-07-10",
        ]),
    })
    rng = np.random.default_rng(42)

    negs = sample_negatives(
        registrations=registrations,
        events=events,
        n_per_positive=3,
        hard_fraction=0.3,
        rng=rng,
    )
    assert all(n["event_id"] not in (10, 50) for _, n in negs.iterrows())
    assert len(negs) == 3


def test_sample_negatives_none_when_no_candidates():
    registrations = pd.DataFrame({
        "user_id": [1], "event_id": [10],
        "created_at": pd.to_datetime(["2025-06-01"]),
    })
    events = pd.DataFrame({"id": [10], "country_code": ["PT"],
                           "published_at": pd.to_datetime(["2025-01-01"]),
                           "start_at": pd.to_datetime(["2025-06-10"])})
    negs = sample_negatives(
        registrations=registrations, events=events,
        n_per_positive=5, hard_fraction=0.3,
        rng=np.random.default_rng(0),
    )
    assert len(negs) == 0


def test_sample_negatives_does_not_exclude_a_later_registration():
    registrations = pd.DataFrame({
        "user_id": [1, 1],
        "event_id": [10, 20],
        "created_at": pd.to_datetime(["2025-06-01", "2025-06-15"]),
    })
    events = pd.DataFrame({
        "id": [10, 20],
        "country_code": ["PT", "PT"],
        "published_at": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "start_at": pd.to_datetime(["2025-06-10", "2025-06-20"]),
    })

    negs = sample_negatives(
        registrations=registrations,
        events=events,
        n_per_positive=1,
        hard_fraction=0.0,
        rng=np.random.default_rng(0),
    )

    june_query = negs[negs["created_at"] == pd.Timestamp("2025-06-01")]
    assert june_query["event_id"].tolist() == [20]
