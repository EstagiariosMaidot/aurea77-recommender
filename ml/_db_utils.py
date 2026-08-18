"""Shared low-level helpers for DB backend detection.

Extracted from ``ml/data_loader.py`` and ``ml/feed_data_loader.py`` to avoid
having two identical private definitions of the same helper.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def is_sqlite(conn: Any) -> bool:
    """Return True when ``conn`` is a raw :class:`sqlite3.Connection`.

    Used by data loaders to pick the correct SQL placeholder (``?`` for SQLite,
    ``%s`` for pymysql-backed connections).
    """
    return isinstance(conn, sqlite3.Connection)
