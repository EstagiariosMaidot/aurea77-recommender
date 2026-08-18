from pathlib import Path

import numpy as np

from ml.artifacts import (
    ArtifactStore,
    remove_event_embedding,
    save_artifacts,
    upsert_event_embedding,
)


class _DummyModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=np.float32)


def test_save_and_load_roundtrip(tmp_path: Path):
    emb = np.random.default_rng(0).standard_normal((4, 384)).astype(np.float32)
    ids = [10, 20, 30, 40]
    metadata = {
        "trained_at": "2026-04-16T10:00:00Z",
        "feature_names": ["sem_sim"],
        "model_version": "v1-test",
    }
    save_artifacts(
        tmp_path,
        model=_DummyModel(),
        event_embeddings=emb,
        event_ids=ids,
        metadata=metadata,
    )
    store = ArtifactStore.load(tmp_path)
    assert store.metadata["model_version"] == "v1-test"
    assert store.event_embeddings.shape == (4, 384)
    assert store.event_id_to_row == {10: 0, 20: 1, 30: 2, 40: 3}
    assert store.model.predict([[0]]).shape == (1,)


def test_load_raises_file_not_found(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        ArtifactStore.load(tmp_path / "does-not-exist")


def _setup_artifacts(tmp_path: Path) -> tuple[np.ndarray, list[int]]:
    """Helper: save artifacts with 3 events and return (embeddings, ids)."""
    emb = np.random.default_rng(1).standard_normal((3, 384)).astype(np.float32)
    ids = [100, 200, 300]
    save_artifacts(
        tmp_path,
        model=_DummyModel(),
        event_embeddings=emb,
        event_ids=ids,
        metadata={"model_version": "test"},
    )
    return emb, ids


def test_upsert_new_event(tmp_path: Path):
    emb, _ = _setup_artifacts(tmp_path)
    new_emb = np.ones(384, dtype=np.float32)

    upsert_event_embedding(tmp_path, event_id=999, embedding=new_emb)

    store = ArtifactStore.load(tmp_path)
    assert store.event_embeddings.shape == (4, 384)
    assert store.event_id_to_row[999] == 3
    assert np.allclose(store.event_embeddings[3], new_emb)


def test_upsert_existing_event_updates_in_place(tmp_path: Path):
    _setup_artifacts(tmp_path)
    updated_emb = np.full(384, 0.5, dtype=np.float32)

    upsert_event_embedding(tmp_path, event_id=200, embedding=updated_emb)

    store = ArtifactStore.load(tmp_path)
    assert store.event_embeddings.shape == (3, 384)  # no new row
    assert np.allclose(store.event_embeddings[1], updated_emb)


def test_remove_event_embedding(tmp_path: Path):
    _setup_artifacts(tmp_path)

    removed = remove_event_embedding(tmp_path, event_id=200)

    assert removed is True
    store = ArtifactStore.load(tmp_path)
    assert store.event_embeddings.shape == (2, 384)
    assert 200 not in store.event_id_to_row
    # Remaining events should have correct indices
    assert store.event_id_to_row[100] == 0
    assert store.event_id_to_row[300] == 1


def test_remove_nonexistent_event_returns_false(tmp_path: Path):
    _setup_artifacts(tmp_path)
    assert remove_event_embedding(tmp_path, event_id=999) is False
