"""Database access for the Aurea77 recommendation service.

Supports two backends via the ``DATABASE_URL`` environment variable:

* **SQLite** (development):
  ``sqlite:///path/to/database.sqlite`` or left empty (falls back to the
  sibling ``aurea77-api/database/database.sqlite``).

* **MySQL** (production):
  ``mysql://user:password@host:port/dbname``

The module never writes — SQLite is opened in read-only mode, and the MySQL
user should only have SELECT privileges.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "aurea77-api"
    / "database"
    / "database.sqlite"
)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _parse_database_url(url: str) -> dict[str, Any]:
    """Parse DATABASE_URL into a dict with backend type and connection info."""
    if not url and os.environ.get("DB_CONNECTION") in {"mysql", "mariadb"}:
        return {
            "backend": "mysql",
            "host": os.environ.get("DB_HOST", "127.0.0.1"),
            "port": int(os.environ.get("DB_PORT", "3306")),
            "user": os.environ.get("DB_USERNAME", "root"),
            "password": os.environ.get("DB_PASSWORD", ""),
            "database": os.environ.get("DB_DATABASE", ""),
        }

    if not url or url.startswith("sqlite"):
        # sqlite:///path/to/file.sqlite  or  empty (default)
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
        elif url.startswith("sqlite://"):
            path = url[len("sqlite://"):]
        else:
            path = str(DEFAULT_DB_PATH)
        return {"backend": "sqlite", "path": Path(path)}

    if url.startswith("mysql://") or url.startswith("mysql+pymysql://"):
        parsed = urlparse(url)
        return {
            "backend": "mysql",
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": parsed.username or "root",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/"),
        }

    raise ValueError(f"Unsupported DATABASE_URL scheme: {url.split('://')[0]}")


_config = _parse_database_url(DATABASE_URL)


def open_read_connection() -> Any:
    """Open a read-only database connection owned by the caller."""
    if _config["backend"] == "sqlite":
        db_path: Path = _config["path"]
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found at {db_path}")
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    else:
        import pymysql

        return pymysql.connect(
            host=_config["host"],
            port=_config["port"],
            user=_config["user"],
            password=_config["password"],
            database=_config["database"],
            charset="utf8mb4",
            read_timeout=30,
            connect_timeout=10,
        )


@contextmanager
def get_connection() -> Iterator[Any]:
    """Yield a read-only database connection and close it on exit."""
    conn = open_read_connection()
    try:
        yield conn
    finally:
        conn.close()


def is_mysql() -> bool:
    """Return True if the configured backend is MySQL."""
    return _config["backend"] == "mysql"
