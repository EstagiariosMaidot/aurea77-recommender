# tests/test_metrics.py
import numpy as np

from evaluate import ndcg_at_k, map_at_k, hit_at_k, coverage_at_k


def test_ndcg_perfect_ranking_is_one():
    ranked = [1, 2, 3]
    truth  = {1}
    assert abs(ndcg_at_k(ranked, truth, k=3) - 1.0) < 1e-9


def test_ndcg_irrelevant_ranking_is_zero():
    assert ndcg_at_k([9, 8, 7], {1}, k=3) == 0.0


def test_hit_at_k_positive_and_negative():
    assert hit_at_k([1, 2], {1}, k=2) == 1
    assert hit_at_k([3, 4], {1}, k=2) == 0


def test_map_at_k_simple_case():
    assert abs(map_at_k([1, 9, 2], {1, 2}, k=3) - (1 + 2 / 3) / 2) < 1e-9


def test_coverage_at_k():
    top_k_per_user = {
        1: [10, 20],
        2: [20, 30],
        3: [40, 50],
    }
    catalog = {10, 20, 30, 40, 50, 60}
    assert abs(coverage_at_k(top_k_per_user, catalog, k=2) - 5 / 6) < 1e-9
