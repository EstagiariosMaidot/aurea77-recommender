"""Lightweight in-memory metrics collector for the recommendation service.

Thread-safe counters, gauges and histograms — no external dependencies.
Metrics are exposed via the ``/metrics`` endpoint as JSON.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


_START_TIME = time.monotonic()


def uptime_seconds() -> float:
    return time.monotonic() - _START_TIME


@dataclass
class _HistogramBucket:
    """Accumulates values for a named metric."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def snapshot(self) -> dict:
        if self.count == 0:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "total": 0}
        return {
            "count": self.count,
            "avg": round(self.total / self.count, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "total": round(self.total, 4),
        }


class MetricsCollector:
    """Global, thread-safe metrics store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, _HistogramBucket] = defaultdict(_HistogramBucket)

    # -- primitives --

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].observe(value)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Context manager that records elapsed seconds into a histogram."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - t0)

    # -- snapshot --

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_seconds": round(uptime_seconds(), 1),
                "counters": dict(self._counters),
                "gauges": {k: round(v, 4) for k, v in self._gauges.items()},
                "histograms": {
                    k: v.snapshot() for k, v in self._histograms.items()
                },
            }


# Singleton used by the entire application
metrics = MetricsCollector()
