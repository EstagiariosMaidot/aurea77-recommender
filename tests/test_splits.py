# tests/test_splits.py
from datetime import datetime

import pandas as pd

from ml.splits import temporal_cutoff, temporal_split


def test_temporal_cutoff_at_percentile_80():
    dates = pd.Series(pd.to_datetime([
        "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01",
    ]))
    c = temporal_cutoff(dates, percentile=0.80)
    assert datetime(2025, 4, 1) <= c <= datetime(2025, 5, 2)


def test_temporal_split_buckets_rows_correctly():
    reg = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "event_id": [10, 20, 30, 40],
        "registered_at": pd.to_datetime([
            "2025-01-01", "2025-06-01", "2025-03-01", "2025-12-01",
        ]),
    })
    cutoff = datetime(2025, 7, 1)
    train, test = temporal_split(reg, cutoff=cutoff, date_col="registered_at")
    assert set(train["event_id"]) == {10, 20, 30}
    assert set(test["event_id"]) == {40}
