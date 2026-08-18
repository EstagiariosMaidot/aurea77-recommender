"""Read-only loaders that convert the Aurea77 database into pandas DataFrames.

Works with both SQLite (development) and MySQL (production) connections.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ml._db_utils import is_sqlite as _is_sqlite


# Registration statuses have existed under more than one name in production.
# Keep the normalisation at the ingestion boundary so training and serving make
# exactly the same decision about which interactions signal interest.
REGISTRATION_STATUS_MAP: dict[str, str] = {
    "registered": "registered",
    "inscribed": "registered",
    "planned": "planned",
    "follow": "planned",
    "to register": "planned",
    "completed": "completed",
    "finished": "completed",
}
INTERESTED_STATUSES: tuple[str, ...] = tuple(sorted(set(REGISTRATION_STATUS_MAP.values())))


def load_events(conn: Any) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
            id, name, description, start_at, end_at,
            country_code, country, city, latitude, longitude,
            published_at, organizer_id, organizer_organization_id, slug
        FROM events
        """,
        conn,
        parse_dates=["start_at", "end_at", "published_at"],
    )
    df["country_code"] = df["country_code"].fillna(df["country"])
    return df


def load_registrations(conn: Any) -> pd.DataFrame:
    # ``created_at`` is the canonical timestamp for a registration.  The old
    # ``registered_at`` field is incomplete in the source data and must not be
    # used for temporal splits, candidate validity, or inference cut-offs.
    raw_statuses = tuple(REGISTRATION_STATUS_MAP)
    placeholder = "?" if _is_sqlite(conn) else "%s"
    placeholders = ",".join([placeholder] * len(raw_statuses))
    df = pd.read_sql_query(
        f"""
        SELECT user_id, event_id, status, created_at
        FROM event_registrations
        WHERE LOWER(status) IN ({placeholders})
        """,
        conn,
        params=raw_statuses,
        parse_dates=["created_at"],
    )
    df["status"] = df["status"].str.strip().str.lower().map(REGISTRATION_STATUS_MAP)
    df = df[df["status"].isin(INTERESTED_STATUSES)].copy()
    return df


def load_reviews(conn: Any) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT user_id, event_id, score, created_at
        FROM event_reviews
        WHERE trust_status != 2
        """,
        conn,
        parse_dates=["created_at"],
    )


def load_athlete_profiles(conn: Any) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT user_id, nationality, sports_practiced FROM athlete_profiles",
        conn,
    )

    def parse_sports(raw: str | None) -> list[str]:
        if not raw:
            return []
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return v if isinstance(v, list) else []

    df["sports_practiced"] = df["sports_practiced"].map(parse_sports)
    return df


def load_category_map(
    conn: Any, *, include_slugs: bool = False
) -> dict[int, set[int]] | tuple[dict[int, set[int]], dict[int, str]]:
    rows = pd.read_sql_query(
        "SELECT event_id, category_id FROM category_event", conn
    ).values.tolist()
    mapping: dict[int, set[int]] = {}
    for event_id, category_id in rows:
        mapping.setdefault(int(event_id), set()).add(int(category_id))

    if not include_slugs:
        return mapping

    slug_rows = pd.read_sql_query(
        "SELECT id, slug FROM categories", conn
    ).values.tolist()
    slug_by_id = {int(r[0]): str(r[1]) for r in slug_rows}
    return mapping, slug_by_id
