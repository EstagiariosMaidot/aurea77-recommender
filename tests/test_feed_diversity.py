"""Tests for feed diversity guardrails."""

from __future__ import annotations

import pytest

from ml.feed_diversity import enforce_source_floor, cap_per_author


def test_enforce_source_floor_pulls_up_discovery_when_below_floor():
    # 10 slots, model ranked 9 friends then 1 discovery — floor 30% means we need 3 discovery in top-10.
    ranked = list(range(1, 11))  # 1..10
    sources = {1: "friend", 2: "friend", 3: "friend", 4: "friend", 5: "friend",
               6: "friend", 7: "friend", 8: "friend", 9: "friend", 10: "discovery"}
    # Also give the sampler a couple of discovery reserves not in the top-10 initially:
    reserves = [11, 12, 13]
    sources_full = {**sources, 11: "discovery", 12: "discovery", 13: "discovery"}
    result = enforce_source_floor(
        ranked_ids=ranked,
        reserves=reserves,
        sources=sources_full,
        source="discovery",
        floor_ratio=0.30,
    )
    # Result contains at least 3 discovery items (floor of 30% × 10 = 3).
    discovery_in_top = sum(1 for pid in result[:10] if sources_full[pid] == "discovery")
    assert discovery_in_top >= 3
    # Total length unchanged (10).
    assert len(result) == 10
    # Order preserves the model's ranking where possible.
    friends_kept = [pid for pid in result if sources_full[pid] == "friend"]
    assert friends_kept == sorted(friends_kept)


def test_enforce_source_floor_noop_when_already_above():
    ranked = [1, 2, 3, 4, 5]
    sources = {1: "friend", 2: "discovery", 3: "friend", 4: "discovery", 5: "friend"}
    result = enforce_source_floor(
        ranked_ids=ranked, reserves=[], sources=sources,
        source="discovery", floor_ratio=0.20,
    )
    # Already 2/5 = 40% discovery, floor is 20%. Result is unchanged.
    assert result == ranked


def test_cap_per_author_removes_extras_beyond_cap():
    ranked = [1, 2, 3, 4, 5, 6]
    authors = {1: 100, 2: 100, 3: 100, 4: 200, 5: 300, 6: 100}  # 4 posts by author 100
    result = cap_per_author(ranked, authors=authors, cap=2)
    # Only 2 posts by author 100 should remain: 1 and 2 (earliest in ranking).
    remaining_100 = [pid for pid in result if authors[pid] == 100]
    assert remaining_100 == [1, 2]
    # Other authors unaffected.
    assert 4 in result and 5 in result


def test_cap_per_author_with_cap_1_produces_one_per_author():
    ranked = [1, 2, 3, 4]
    authors = {1: 100, 2: 100, 3: 200, 4: 200}
    result = cap_per_author(ranked, authors=authors, cap=1)
    assert result == [1, 3]


def test_cap_per_author_raises_when_cap_below_1():
    with pytest.raises(ValueError):
        cap_per_author([1], authors={1: 100}, cap=0)
