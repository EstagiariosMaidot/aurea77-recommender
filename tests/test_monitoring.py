"""Tests for the in-memory metrics collector."""

from __future__ import annotations

import threading

from monitoring import MetricsCollector


def test_counter_increments():
    m = MetricsCollector()
    m.inc("requests")
    m.inc("requests")
    m.inc("requests", 3)
    snap = m.snapshot()
    assert snap["counters"]["requests"] == 5


def test_gauge_sets_value():
    m = MetricsCollector()
    m.gauge("temperature", 36.6)
    m.gauge("temperature", 37.0)
    snap = m.snapshot()
    assert snap["gauges"]["temperature"] == 37.0


def test_histogram_observe():
    m = MetricsCollector()
    m.observe("latency", 0.1)
    m.observe("latency", 0.5)
    m.observe("latency", 0.2)
    snap = m.snapshot()
    h = snap["histograms"]["latency"]
    assert h["count"] == 3
    assert h["min"] == 0.1
    assert h["max"] == 0.5
    assert abs(h["avg"] - 0.2667) < 0.01


def test_timer_context_manager():
    m = MetricsCollector()
    with m.timer("op"):
        pass  # near-zero elapsed
    snap = m.snapshot()
    h = snap["histograms"]["op"]
    assert h["count"] == 1
    assert h["avg"] >= 0


def test_snapshot_includes_uptime():
    m = MetricsCollector()
    snap = m.snapshot()
    assert "uptime_seconds" in snap
    assert snap["uptime_seconds"] >= 0


def test_thread_safety():
    m = MetricsCollector()

    def worker():
        for _ in range(1000):
            m.inc("concurrent")
            m.observe("concurrent_hist", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = m.snapshot()
    assert snap["counters"]["concurrent"] == 4000
    assert snap["histograms"]["concurrent_hist"]["count"] == 4000


def test_empty_histogram_snapshot():
    m = MetricsCollector()
    m.observe("empty_test", 0)  # ensure bucket exists but test default
    # Also check a histogram that was never observed
    snap = m.snapshot()
    assert "never_seen" not in snap["histograms"]
