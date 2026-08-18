"""Build a tiny SQLite DB with feed_posts and social data for testing feed loaders."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE feed_posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    body TEXT,
    privacy TEXT NOT NULL DEFAULT 'public',
    likes_count INTEGER NOT NULL DEFAULT 0,
    comments_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE feed_post_likes (
    id INTEGER PRIMARY KEY,
    feed_post_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    reaction_type TEXT NOT NULL DEFAULT 'like',
    created_at TEXT
);

CREATE TABLE feed_post_comments (
    id INTEGER PRIMARY KEY,
    feed_post_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE post_media (
    id INTEGER PRIMARY KEY,
    post_id INTEGER NOT NULL,
    media_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE friendships (
    id INTEGER PRIMARY KEY,
    requester_id INTEGER NOT NULL,
    receiver_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    action_by_id INTEGER,
    created_at TEXT
);

CREATE TABLE user_blocks (
    id INTEGER PRIMARY KEY,
    blocker_id INTEGER NOT NULL,
    blocked_id INTEGER NOT NULL,
    created_at TEXT
);

CREATE TABLE user_mutes (
    id INTEGER PRIMARY KEY,
    muter_id INTEGER NOT NULL,
    muted_id INTEGER NOT NULL,
    created_at TEXT
);

CREATE TABLE post_impressions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    feed_post_id INTEGER NOT NULL,
    served_at TEXT NOT NULL,
    source TEXT NOT NULL,
    position_in_feed INTEGER NOT NULL,
    model_version TEXT,
    viewed_for_ms INTEGER,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE feed_post_interactions_log (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    feed_post_id INTEGER NOT NULL,
    interaction_type TEXT NOT NULL,
    reaction_type TEXT,
    occurred_at TEXT NOT NULL,
    source TEXT,
    model_version TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""


def build_tiny_feed_db(path: Path) -> None:
    """Populate a tiny SQLite DB with a deterministic dataset for feed tests.

    Users:      1..4 (4 users)
    Posts:      1..6 (users 1,2,3 authored posts; user 4 authored none)
    Friendships: 1-2 accepted, 1-3 pending
    Likes:      user 2 liked posts 1 and 3; user 4 liked post 1
    Comments:   user 3 commented on post 1
    Impressions: user 2 has 3 impressions from friend source
    Interactions log: seeded from likes/comments
    """
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)

    now = datetime(2026, 7, 10, 12, 0, 0)
    iso = lambda dt: dt.isoformat(sep=" ")

    conn.executemany(
        "INSERT INTO users(id,email,created_at) VALUES (?,?,?)",
        [
            (1, "alice@test.example", iso(now - timedelta(days=200))),
            (2, "bob@test.example", iso(now - timedelta(days=180))),
            (3, "carol@test.example", iso(now - timedelta(days=90))),
            (4, "dave@test.example", iso(now - timedelta(days=10))),
        ],
    )

    conn.executemany(
        "INSERT INTO feed_posts(id,user_id,body,privacy,likes_count,comments_count,created_at) VALUES (?,?,?,?,?,?,?)",
        [
            (1, 1, "First post by Alice about trail running.", "public", 2, 1, iso(now - timedelta(days=5))),
            (2, 1, "Second Alice post — short one.", "public", 0, 0, iso(now - timedelta(days=3))),
            (3, 2, "Bob shares a marathon result.", "public", 1, 0, iso(now - timedelta(days=4))),
            (4, 2, "Bob posts a photo.", "public", 0, 0, iso(now - timedelta(days=2))),
            (5, 3, "Carol talks about cycling.", "public", 0, 0, iso(now - timedelta(days=1))),
            (6, 3, "Carol private post.", "friends", 0, 0, iso(now - timedelta(hours=6))),
        ],
    )

    conn.executemany(
        "INSERT INTO post_media(id,post_id,media_path,media_type,sort_order) VALUES (?,?,?,?,?)",
        [
            (1, 4, "/media/bob-photo.jpg", "image", 0),
            (2, 5, "/media/carol-video.mp4", "video", 0),
        ],
    )

    conn.executemany(
        "INSERT INTO feed_post_likes(id,feed_post_id,user_id,reaction_type,created_at) VALUES (?,?,?,?,?)",
        [
            (1, 1, 2, "like", iso(now - timedelta(days=4))),
            (2, 1, 4, "love", iso(now - timedelta(days=3))),
            (3, 3, 2, "like", iso(now - timedelta(days=3))),
        ],
    )

    conn.executemany(
        "INSERT INTO feed_post_comments(id,feed_post_id,user_id,body,created_at) VALUES (?,?,?,?,?)",
        [
            (1, 1, 3, "Nice!", iso(now - timedelta(days=4))),
        ],
    )

    conn.executemany(
        "INSERT INTO friendships(id,requester_id,receiver_id,status,action_by_id,created_at) VALUES (?,?,?,?,?,?)",
        [
            (1, 1, 2, "accepted", 2, iso(now - timedelta(days=150))),
            (2, 1, 3, "pending", None, iso(now - timedelta(days=20))),
        ],
    )

    conn.executemany(
        "INSERT INTO post_impressions(id,user_id,feed_post_id,served_at,source,position_in_feed,model_version,viewed_for_ms) VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, 2, 1, iso(now - timedelta(hours=48)), "friend", 0, None, 3200),
            (2, 2, 2, iso(now - timedelta(hours=48)), "friend", 1, None, 1200),
            (3, 2, 3, iso(now - timedelta(hours=24)), "friend", 0, None, 5000),
        ],
    )

    conn.executemany(
        "INSERT INTO feed_post_interactions_log(id,user_id,feed_post_id,interaction_type,reaction_type,occurred_at,source,model_version) VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, 2, 1, "like", "like", iso(now - timedelta(days=4)), "friend", None),
            (2, 4, 1, "like", "love", iso(now - timedelta(days=3)), None, None),
            (3, 3, 1, "comment", None, iso(now - timedelta(days=4)), None, None),
            (4, 2, 3, "like", "like", iso(now - timedelta(days=3)), "friend", None),
        ],
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    import sys
    build_tiny_feed_db(Path(sys.argv[1] if len(sys.argv) > 1 else "tiny_feed.sqlite"))
    print("Built tiny feed DB.")
