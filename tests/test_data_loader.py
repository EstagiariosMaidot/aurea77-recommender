# tests/test_data_loader.py
import pandas as pd

from ml.data_loader import (
    load_events,
    load_registrations,
    load_reviews,
    load_athlete_profiles,
    load_category_map,
)


def test_load_events_returns_all_events(tiny_db):
    df = load_events(tiny_db)
    assert len(df) == 20
    assert set(df.columns) >= {
        "id", "name", "description", "start_at", "country_code",
        "city", "latitude", "longitude", "published_at", "organizer_id",
    }
    assert pd.api.types.is_datetime64_any_dtype(df["start_at"])


def test_load_registrations_only_interested_statuses(tiny_db):
    df = load_registrations(tiny_db)
    assert (df["status"].isin({"registered", "planned", "completed"})).all()
    assert "finished" not in set(df["status"])
    assert "completed" in set(df["status"])
    assert set(df.columns) >= {"user_id", "event_id", "status", "created_at"}
    assert "registered_at" not in df.columns


def test_load_reviews(tiny_db):
    df = load_reviews(tiny_db)
    assert set(df.columns) >= {"user_id", "event_id", "score"}
    assert (df["score"].between(1, 5)).all()


def test_load_athlete_profiles_parses_sports_json(tiny_db):
    df = load_athlete_profiles(tiny_db)
    alice = df.loc[df["user_id"] == 1].iloc[0]
    assert alice["sports_practiced"] == ["trail_running", "ultra"]
    eva = df.loc[df["user_id"] == 5].iloc[0]
    assert eva["sports_practiced"] == []


def test_load_category_map(tiny_db):
    mapping = load_category_map(tiny_db)
    assert 1 in mapping
    assert isinstance(mapping[1], set)
    slugs = load_category_map(tiny_db, include_slugs=True)[1]
    assert slugs == {
        1: "trail-running", 2: "road-running", 3: "ultra", 4: "cycling",
    }
