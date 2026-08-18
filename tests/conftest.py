"""Shared pytest fixtures for the recommendation tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.build_tiny_db import build


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch):
    monkeypatch.setenv("RECOMMENDER_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _default_jwt_secret(monkeypatch):
    monkeypatch.setenv("RECOMMENDER_JWT_SECRET", "test-jwt-secret")


@pytest.fixture(autouse=True)
def _testclient_sends_api_key(monkeypatch):
    """Ensure every TestClient sends X-API-Key by default — protected
    endpoints require it now that dev-mode fallback is gone."""
    from fastapi.testclient import TestClient

    original_init = TestClient.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.headers["X-API-Key"] = "test-key"

    monkeypatch.setattr(TestClient, "__init__", patched_init)


@pytest.fixture(scope="session")
def tiny_db_path(tmp_path_factory) -> Path:
    """Build a fresh tiny SQLite once per test session."""
    target = tmp_path_factory.mktemp("db") / "tiny.sqlite"
    build(target)
    return target


@pytest.fixture
def tiny_db(tiny_db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection to the tiny fixture DB."""
    uri = f"file:{tiny_db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
