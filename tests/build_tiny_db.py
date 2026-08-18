"""Materialize a tiny SQLite fixture mirroring the shape of the Laravel DB.

Run with: python -m tests.build_tiny_db

Creates tests/fixtures/tiny.sqlite with 5 users, 20 events across 4 categories,
a realistic mix of registrations, reviews, and athlete profiles.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


SCHEMA = """
CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);

CREATE TABLE athlete_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    nationality TEXT,
    sports_practiced TEXT  -- JSON array
);

CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL
);

CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    start_at TEXT NOT NULL,
    end_at TEXT,
    location_text TEXT,
    country TEXT,
    country_code TEXT,
    city TEXT,
    latitude REAL,
    longitude REAL,
    published_at TEXT,
    organizer_id INTEGER,
    organizer_organization_id INTEGER,
    slug TEXT
);

CREATE TABLE category_event (
    event_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL
);

CREATE TABLE event_registrations (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    registered_at TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE event_reviews (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    score INTEGER NOT NULL,
    trust_status INTEGER DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);
"""


def build(out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(out_path)
    conn.executescript(SCHEMA)

    now = datetime(2026, 4, 16, 12, 0, 0)

    # 5 users with profiles
    users = [
        (1, "alice@test", "PT", '["trail_running", "ultra"]'),
        (2, "bruno@test", "PT", '["road_running"]'),
        (3, "carla@test", "ES", '["trail_running"]'),
        (4, "david@test", "PT", '["cycling"]'),
        (5, "eva@test",   "PT", None),              # cold-start: no sports declared
    ]
    for uid, email, nat, sports in users:
        conn.execute("INSERT INTO users(id,email) VALUES (?,?)", (uid, email))
        conn.execute(
            "INSERT INTO athlete_profiles(id,user_id,nationality,sports_practiced)"
            " VALUES (?,?,?,?)",
            (uid, uid, nat, sports),
        )

    # 4 categories
    categories = [
        (1, "Trail Running", "trail-running"),
        (2, "Road Running",  "road-running"),
        (3, "Ultra",         "ultra"),
        (4, "Cycling",       "cycling"),
    ]
    conn.executemany(
        "INSERT INTO categories(id,name,slug) VALUES (?,?,?)", categories
    )

    # 20 events — half in the past (for training), half in the future (candidates)
    # locations cluster around Lisbon (38.7, -9.1), Porto (41.15, -8.6), Madrid (40.4, -3.7)
    events = []
    cat_links = []
    clusters = [
        ("PT", "Lisboa", 38.7, -9.1),
        ("PT", "Porto",  41.15, -8.6),
        ("ES", "Madrid", 40.4, -3.7),
    ]
    for i in range(1, 21):
        offset_days = -60 + (i * 15)      # spans -45d .. +240d from `now`
        start = now + timedelta(days=offset_days)
        cc_name, city, lat, lng = clusters[i % 3]
        cats = [(i % 4) + 1]              # at least one cat; deterministic
        if i % 5 == 0:
            cats.append((cats[0] % 4) + 1)  # some events have 2 cats
        events.append(
            (
                i,
                f"Event {i}",
                f"Nice event number {i} in {city}, focusing on endurance.",
                start.isoformat(sep=" "),
                (start + timedelta(hours=4)).isoformat(sep=" "),
                f"{city}, {cc_name}",
                cc_name,          # `country`   (original column)
                cc_name,           # `country_code` (geocoding column; same value here)
                city,
                lat,
                lng,
                (start - timedelta(days=30)).isoformat(sep=" "),
                1,
                None,
                f"event-{i}",
            )
        )
        for c in cats:
            cat_links.append((i, c))

    conn.executemany(
        """
        INSERT INTO events(
          id,name,description,start_at,end_at,location_text,country,country_code,city,
          latitude,longitude,published_at,organizer_id,organizer_organization_id,slug
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        events,
    )
    conn.executemany(
        "INSERT INTO category_event(event_id,category_id) VALUES (?,?)", cat_links
    )

    # Registrations: alice loves trail + ultra (events with cat 1 and 3),
    # bruno goes to road running (cat 2), carla goes to trail (cat 1),
    # david goes to cycling (cat 4). eva has zero history.
    registrations = []
    reg_id = 1
    preferences = {1: {1, 3}, 2: {2}, 3: {1}, 4: {4}}
    for uid, prefs in preferences.items():
        for event_id, cat_id in cat_links:
            if cat_id in prefs:
                # only register for past events (start_at < now)
                start_iso = [e[3] for e in events if e[0] == event_id][0]
                if start_iso < now.isoformat(sep=" "):
                    reg_at = (now - timedelta(days=10)).isoformat(sep=" ")
                    # ``finished`` is a legacy production status. The loader
                    # must normalise it to the canonical ``completed`` value.
                    status = "finished" if reg_id == 1 else "completed"
                    registrations.append(
                        (reg_id, uid, event_id, status, reg_at, reg_at, reg_at)
                    )
                    reg_id += 1
    conn.executemany(
        """
        INSERT INTO event_registrations(
          id,user_id,event_id,status,registered_at,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?)
        """,
        registrations,
    )

    # Reviews: alice gives high scores, bruno mixed
    review_id = 1
    review_rows = []
    for r in registrations:
        uid, event_id = r[1], r[2]
        if uid == 1:
            score = 5
        elif uid == 2:
            score = 3
        elif uid == 3:
            score = 4
        else:
            continue  # david doesn't review
        review_rows.append(
            (review_id, event_id, uid, score, 1,
             (now - timedelta(days=5)).isoformat(sep=" "),
             (now - timedelta(days=5)).isoformat(sep=" "))
        )
        review_id += 1
    conn.executemany(
        """
        INSERT INTO event_reviews(
          id,event_id,user_id,score,trust_status,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?)
        """,
        review_rows,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "fixtures" / "tiny.sqlite"
    build(target)
    print(f"Wrote {target}")
