"""Loaders that read the Aurea77 feed/social tables into pandas DataFrames.

Works with both SQLite (development/testing) and MySQL (production) connections.
Read-only — the loaders never write.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ml._db_utils import is_sqlite as _is_sqlite


_IMPRESSION_COLUMNS = [
    "user_id", "feed_post_id", "served_at", "source",
    "position_in_feed", "model_version", "viewed_for_ms",
]
_INTERACTION_COLUMNS = [
    "user_id", "feed_post_id", "interaction_type", "reaction_type",
    "occurred_at", "source", "model_version",
]


def _is_missing_table_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "no such table" in message or "doesn't exist" in message or "does not exist" in message


def _empty_optional_table(columns: list[str], datetime_column: str) -> pd.DataFrame:
    data: dict[str, pd.Series] = {
        column: pd.Series(dtype="datetime64[ns]" if column == datetime_column else "object")
        for column in columns
    }
    return pd.DataFrame(data)


def _table_columns(conn: Any, table: str) -> set[str]:
    """Return table columns for both Laravel/MySQL and SQLite test schemas."""
    if _is_sqlite(conn):
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}
    rows = pd.read_sql_query(f"SHOW COLUMNS FROM `{table}`", conn)
    return set(rows["Field"].astype(str))


def load_feed_posts(conn: Any) -> pd.DataFrame:
    """Return every feed_post as a DataFrame with parsed timestamps."""
    df = pd.read_sql_query(
        """
        SELECT
            id, user_id, body, privacy,
            likes_count, comments_count,
            created_at, updated_at
        FROM feed_posts
        """,
        conn,
        parse_dates=["created_at", "updated_at"],
    )
    return df


def load_users_min(conn: Any) -> pd.DataFrame:
    """Return the minimal user projection used by the feed features (id, created_at)."""
    df = pd.read_sql_query(
        """
        SELECT id, created_at
        FROM users
        """,
        conn,
        parse_dates=["created_at"],
    )
    return df


def load_friendships(conn: Any, *, include_pending: bool = False) -> pd.DataFrame:
    """Return the friendships table filtered by status.

    By default only `accepted` friendships are returned (which drive the friend
    candidate pool). Pass ``include_pending=True`` to also include pending rows.
    """
    if include_pending:
        query = """
            SELECT id, requester_id, receiver_id, status, created_at
            FROM friendships
            WHERE status IN ('accepted', 'pending')
        """
    else:
        query = """
            SELECT id, requester_id, receiver_id, status, created_at
            FROM friendships
            WHERE status = 'accepted'
        """
    return pd.read_sql_query(query, conn, parse_dates=["created_at"])


def load_post_media(conn: Any) -> pd.DataFrame:
    """Return the post_media rows with columns post_id, media_type, sort_order."""
    return pd.read_sql_query(
        """
        SELECT post_id, media_type, sort_order
        FROM post_media
        """,
        conn,
    )


def load_user_blocks(conn: Any) -> pd.DataFrame:
    """Return user blocks normalized to ``blocker_id`` and ``blocked_id``."""
    columns = _table_columns(conn, "user_blocks")
    blocker = "blocker_user_id" if "blocker_user_id" in columns else "blocker_id"
    blocked = "blocked_user_id" if "blocked_user_id" in columns else "blocked_id"
    return pd.read_sql_query(
        f"SELECT {blocker} AS blocker_id, {blocked} AS blocked_id FROM user_blocks",
        conn,
    )


def load_user_mutes(conn: Any) -> pd.DataFrame:
    """Return user mutes normalized to ``muter_id`` and ``muted_id``."""
    columns = _table_columns(conn, "user_mutes")
    muter = "muter_user_id" if "muter_user_id" in columns else "muter_id"
    muted = "muted_user_id" if "muted_user_id" in columns else "muted_id"
    return pd.read_sql_query(
        f"SELECT {muter} AS muter_id, {muted} AS muted_id FROM user_mutes",
        conn,
    )


def load_post_impressions(
    conn: Any,
    *,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Return impressions from the post_impressions table.

    When ``since`` is provided, only rows served after that instant are returned.
    Times are compared in the DB's native timezone (UTC).
    """
    try:
        if since is not None:
            params = (since.strftime("%Y-%m-%d %H:%M:%S"),)
            placeholder = "?" if _is_sqlite(conn) else "%s"
            query = f"""
                SELECT
                    user_id, feed_post_id, served_at, source,
                    position_in_feed, model_version, viewed_for_ms
                FROM post_impressions
                WHERE served_at >= {placeholder}
            """
            return pd.read_sql_query(query, conn, params=params, parse_dates=["served_at"])

        return pd.read_sql_query(
            """
            SELECT
                user_id, feed_post_id, served_at, source,
                position_in_feed, model_version, viewed_for_ms
            FROM post_impressions
            """,
            conn,
            parse_dates=["served_at"],
        )
    except Exception as exc:
        if _is_missing_table_error(exc):
            return _empty_optional_table(_IMPRESSION_COLUMNS, "served_at")
        raise


def load_interactions_log(
    conn: Any,
    *,
    interaction_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the denormalized interactions log, optionally filtered by type."""
    try:
        if interaction_types:
            placeholder = "?" if _is_sqlite(conn) else "%s"
            placeholders = ",".join([placeholder] * len(interaction_types))
            query = f"""
                SELECT
                    user_id, feed_post_id, interaction_type, reaction_type,
                    occurred_at, source, model_version
                FROM feed_post_interactions_log
                WHERE interaction_type IN ({placeholders})
            """
            return pd.read_sql_query(
                query,
                conn,
                params=interaction_types,
                parse_dates=["occurred_at"],
            )

        return pd.read_sql_query(
            """
            SELECT
                user_id, feed_post_id, interaction_type, reaction_type,
                occurred_at, source, model_version
            FROM feed_post_interactions_log
            """,
            conn,
            parse_dates=["occurred_at"],
        )
    except Exception as exc:
        if _is_missing_table_error(exc):
            return _empty_optional_table(_INTERACTION_COLUMNS, "occurred_at")
        raise
