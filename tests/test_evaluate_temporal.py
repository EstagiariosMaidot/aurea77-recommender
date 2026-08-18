from __future__ import annotations

import pandas as pd

from evaluate import _candidate_pool


def test_candidate_pool_is_as_of_the_recommendation_time():
    query_at = pd.Timestamp("2025-06-01 12:00:00")
    events = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "published_at": pd.to_datetime([
            "2025-05-01",  # already registered
            "2025-05-01",  # valid
            "2025-06-02",  # not published yet
            "2025-05-01",  # already started
            None,            # unpublished
        ]),
        "start_at": pd.to_datetime([
            "2025-06-10", "2025-06-10", "2025-06-10", "2025-05-30", "2025-06-10",
        ]),
    })

    assert _candidate_pool(
        events,
        registered_event_ids={1},
        query_at=query_at,
    ) == [2]
