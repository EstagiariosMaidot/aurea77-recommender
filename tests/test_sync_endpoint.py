# tests/test_sync_endpoint.py
import subprocess
import sys

from fastapi.testclient import TestClient


def _patch_db(monkeypatch, tiny_db_path):
    """Point the database module at the tiny fixture DB."""
    import database
    monkeypatch.setattr(
        database, "_config",
        {"backend": "sqlite", "path": tiny_db_path},
    )


def test_sync_event_creates_embedding(tmp_path, tiny_db_path, monkeypatch):
    """POST /events/{id}/sync adds the event embedding to artifacts."""
    # Train first to create artifacts
    out = tmp_path / "artifacts"
    subprocess.check_call([
        sys.executable, "-m", "train",
        "--db-path", str(tiny_db_path),
        "--out-dir", str(out),
        "--n-estimators", "50",
        "--seed", "42",
    ])

    # Patch defaults so the app uses our tmp artifacts and DB
    import main
    monkeypatch.setattr(main, "DEFAULT_ARTIFACTS_DIR", out)
    _patch_db(monkeypatch, tiny_db_path)

    from ml.artifacts import ArtifactStore
    store_before = ArtifactStore.load(out)
    n_before = store_before.event_embeddings.shape[0]

    client = TestClient(main.app)

    # Pick an event ID that exists in tiny DB (event 1)
    resp = client.post("/events/1/sync")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["event_id"] == 1
    assert body["action"] == "upserted"

    # Verify embedding was updated (shape should stay same since event 1
    # was already in artifacts from training)
    store_after = ArtifactStore.load(out)
    assert store_after.event_embeddings.shape[0] == n_before
    assert 1 in store_after.event_id_to_row


def test_sync_nonexistent_event_returns_404(tmp_path, tiny_db_path, monkeypatch):
    out = tmp_path / "artifacts"
    subprocess.check_call([
        sys.executable, "-m", "train",
        "--db-path", str(tiny_db_path),
        "--out-dir", str(out),
        "--n-estimators", "50",
        "--seed", "42",
    ])

    import main
    monkeypatch.setattr(main, "DEFAULT_ARTIFACTS_DIR", out)
    _patch_db(monkeypatch, tiny_db_path)

    client = TestClient(main.app)
    resp = client.post("/events/99999/sync")
    assert resp.status_code == 404


def test_sync_without_artifacts_returns_503(tmp_path, tiny_db_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "DEFAULT_ARTIFACTS_DIR", tmp_path / "empty")
    _patch_db(monkeypatch, tiny_db_path)

    client = TestClient(main.app)
    resp = client.post("/events/1/sync")
    assert resp.status_code == 503


def test_delete_event_removes_embedding(tmp_path, tiny_db_path, monkeypatch):
    out = tmp_path / "artifacts"
    subprocess.check_call([
        sys.executable, "-m", "train",
        "--db-path", str(tiny_db_path),
        "--out-dir", str(out),
        "--n-estimators", "50",
        "--seed", "42",
    ])

    import main
    monkeypatch.setattr(main, "DEFAULT_ARTIFACTS_DIR", out)
    _patch_db(monkeypatch, tiny_db_path)

    from ml.artifacts import ArtifactStore
    store_before = ArtifactStore.load(out)
    n_before = store_before.event_embeddings.shape[0]
    # Pick an event that exists in artifacts
    event_id = list(store_before.event_id_to_row.keys())[0]

    client = TestClient(main.app)
    resp = client.delete(f"/events/{event_id}/sync")
    assert resp.status_code == 200
    assert resp.json()["action"] == "removed"

    store_after = ArtifactStore.load(out)
    assert store_after.event_embeddings.shape[0] == n_before - 1
    assert event_id not in store_after.event_id_to_row
