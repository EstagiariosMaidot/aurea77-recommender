"""FastAPI entrypoint for the Aurea77 recommendation service."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Security
from fastapi.security import APIKeyHeader
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse

from rate_limit import limiter


# ── Structured JSON logging ──────────────────────────────────────────
class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _setup_logging() -> None:
    use_json = os.environ.get("LOG_FORMAT", "text") == "json"
    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


_setup_logging()

from auth_jwt import decode_recommender_jwt
from database import get_connection
from ml.artifacts import ArtifactStore, CachedArtifactStore, remove_event_embedding, upsert_event_embedding
from ml.data_loader import load_events
from ml.embeddings import build_event_text, embed_texts
from ml.features import EVENT_TRAINING_SCHEMA_VERSION, FEATURE_NAMES
from monitoring import metrics, uptime_seconds
from recommender import recommend

log = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
API_KEY = os.environ.get("RECOMMENDER_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError(
        "RECOMMENDER_API_KEY is required. Refusing to start in open mode."
    )

# Shared cached store — reloads automatically when files change
_cached_store = CachedArtifactStore(DEFAULT_ARTIFACTS_DIR)

# Lock to serialise writes to the artifacts directory
_sync_lock = threading.Lock()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Reject requests without a valid API key."""
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(
    title="Aurea77 Recommendation Service",
    version="1.0.0",
    description="Hybrid Learning-to-Rank recommender for Aurea77 events.",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429, content={"detail": "rate limit exceeded"}
    ),
)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request count, latency and errors for every endpoint."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0

    path = request.url.path
    metrics.inc(f"http.requests.{path}")
    metrics.observe(f"http.latency_seconds.{path}", elapsed)
    if response.status_code >= 400:
        metrics.inc(f"http.errors.{path}")
    return response


@app.get("/health")
def health() -> dict:
    try:
        store = _cached_store.get()
        meta = store.metadata
        n_embeddings = len(store.event_id_to_row)
    except FileNotFoundError:
        meta = {}
        n_embeddings = 0
    return {
        "status": "ok",
        "uptime_seconds": round(uptime_seconds(), 1),
        "model_version": meta.get("model_version", "unknown"),
        "model_trained_at": meta.get("trained_at", "unknown"),
        "embeddings_count": n_embeddings,
        "event_model_retrain_required": bool(meta) and (
            meta.get("feature_names") != FEATURE_NAMES
            or meta.get("training_schema_version") != EVENT_TRAINING_SCHEMA_VERSION
        ),
    }


@app.get("/metrics", dependencies=[Depends(_verify_api_key)])
def get_metrics() -> dict:
    """Operational metrics snapshot for monitoring dashboards."""
    return metrics.snapshot()


@app.get("/events/embedded-ids", dependencies=[Depends(_verify_api_key)])
def embedded_ids() -> dict:
    """Return all event IDs that have embeddings. Used by the catch-up command."""
    try:
        store = _cached_store.get()
    except FileNotFoundError:
        return {"event_ids": []}
    return {"event_ids": sorted(store.event_id_to_row.keys())}


@app.get("/recommend/{user_id}", dependencies=[Depends(_verify_api_key)])
def recommend_for_user(
    user_id: int,
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    with metrics.timer("recommend.latency_seconds"):
        with get_connection() as conn:
            results = recommend(conn, user_id=user_id, limit=limit,
                                cached_store=_cached_store)

    # Recommendation-specific metrics
    metrics.inc("recommend.total")
    n = len(results)
    if n == 0:
        metrics.inc("recommend.empty")
    else:
        avg_score = sum(r.score for r in results) / n
        metrics.observe("recommend.result_count", n)
        metrics.observe("recommend.avg_score", avg_score)

    try:
        meta = _cached_store.get().metadata
    except FileNotFoundError:
        meta = {}
    return {
        "user_id": user_id,
        "count": n,
        "model_version": meta.get("model_version", "unknown"),
        "recommendations": [
            {"event_id": r.event_id, "score": round(r.score, 4)} for r in results
        ],
    }


@app.post("/events/{event_id}/sync", dependencies=[Depends(_verify_api_key)])
def sync_event(event_id: int) -> dict:
    """Compute (or update) the embedding for a single event.

    Call this from the Laravel API whenever an event is created or edited.
    The event becomes available in recommendations immediately.
    """
    if not (DEFAULT_ARTIFACTS_DIR / "event_embeddings.npy").exists():
        metrics.inc("sync.errors")
        raise HTTPException(
            status_code=503,
            detail="Artifacts not found. Run initial training first.",
        )

    try:
        metadata = ArtifactStore.load(DEFAULT_ARTIFACTS_DIR).metadata
    except FileNotFoundError:
        metadata = {}
    if (
        metadata.get("feature_names") != FEATURE_NAMES
        or metadata.get("training_schema_version") != EVENT_TRAINING_SCHEMA_VERSION
    ):
        metrics.inc("sync.errors")
        raise HTTPException(
            status_code=503,
            detail="Event model uses an outdated feature schema. Retrain before syncing embeddings.",
        )

    with metrics.timer("sync.latency_seconds"):
        with get_connection() as conn:
            events = load_events(conn)
            event_row = events[events["id"] == event_id]
            if event_row.empty:
                metrics.inc("sync.errors")
                raise HTTPException(status_code=404, detail="Event not found")


        texts = build_event_text(event_row)
        embedding = embed_texts(texts)[0]  # single vector (384,)

        with _sync_lock:
            upsert_event_embedding(DEFAULT_ARTIFACTS_DIR, event_id, embedding)
            _cached_store.invalidate()

    metrics.inc("sync.upserted")
    log.info("Synced embedding for event %d", event_id)
    return {"status": "ok", "event_id": event_id, "action": "upserted"}


@app.delete("/events/{event_id}/sync", dependencies=[Depends(_verify_api_key)])
def remove_event(event_id: int) -> dict:
    """Remove an event embedding (e.g. when an event is deleted)."""
    if not (DEFAULT_ARTIFACTS_DIR / "event_embeddings.npy").exists():
        metrics.inc("sync.errors")
        raise HTTPException(
            status_code=503,
            detail="Artifacts not found. Run initial training first.",
        )

    with _sync_lock:
        removed = remove_event_embedding(DEFAULT_ARTIFACTS_DIR, event_id)
        if removed:
            _cached_store.invalidate()

    if not removed:
        metrics.inc("sync.errors")
        raise HTTPException(status_code=404, detail="Event not in embeddings")

    metrics.inc("sync.removed")
    log.info("Removed embedding for event %d", event_id)
    return {"status": "ok", "event_id": event_id, "action": "removed"}


# ── Feed (mixed post) recommender endpoints ──────────────────────────

FEED_ARTIFACTS_DIR = Path(os.environ.get(
    "FEED_ARTIFACTS_DIR",
    str(Path(__file__).resolve().parent / "artifacts" / "feed")
))


@app.get("/posts/embedded-ids", dependencies=[Depends(_verify_api_key)])
def posts_embedded_ids() -> dict:
    id_map_path = FEED_ARTIFACTS_DIR / "post_id_to_row.json"
    if not id_map_path.exists():
        return {"post_ids": []}
    mapping = json.loads(id_map_path.read_text())
    return {"post_ids": sorted(int(k) for k in mapping.keys())}


def _db_path_from_env() -> str | None:
    """Extract a plain file-system path from DATABASE_URL (SQLite only).

    Design note: ``database.py`` reads and caches ``_config`` at import time,
    so ``monkeypatch.setenv("DATABASE_URL", ...) + importlib.reload(main)``
    can leave the underlying config stale. Reading the env var here at
    request time lets tests point the feed endpoints at a fresh SQLite fixture
    without reloading ``database`` too. For MySQL/production this returns
    ``None`` and the caller falls through to the standard ``get_connection()``.
    Follow-up idea: refactor ``database.py`` to lazily resolve config, which
    would let this helper go away entirely.
    """
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return None  # MySQL or empty → let feed_recommender use get_connection()


def _recommend_posts_for_viewer(user_id: int, limit: int) -> dict:
    """Shared feed-serving logic used by both /posts/feed endpoints."""
    from feed_recommender import feed_model_status, recommend_for_viewer

    try:
        ranking_mode, model_version = feed_model_status(str(FEED_ARTIFACTS_DIR))
        recs = recommend_for_viewer(
            viewer_id=user_id,
            db_path=_db_path_from_env(),
            artifacts_dir=str(FEED_ARTIFACTS_DIR),
            limit=limit,
        )
    except Exception:
        log.exception("Feed ranking failed for user %d", user_id)
        raise HTTPException(status_code=503, detail="Feed ranking is temporarily unavailable.")

    return {
        "user_id": user_id,
        "count": len(recs),
        "ranking_mode": ranking_mode,
        "model_version": model_version,
        "recommendations": [
            {"post_id": r.post_id, "source": r.source, "score": round(r.score, 4)}
            for r in recs
        ],
    }


def _extract_user_id_from_bearer(
    authorization: str | None = Header(default=None),
) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        claims = decode_recommender_jwt(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return int(claims["user_id"])


@app.get("/posts/feed/{user_id}", dependencies=[Depends(_verify_api_key)])
def posts_feed(user_id: int, limit: int = Query(10, ge=1, le=50)) -> dict:
    return _recommend_posts_for_viewer(user_id, limit)


@app.post("/posts/feed", dependencies=[Depends(_verify_api_key)])
def posts_feed_authenticated(
    user_id: int = Depends(_extract_user_id_from_bearer),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """New feed endpoint — user_id comes from the signed JWT, not the URL."""
    return _recommend_posts_for_viewer(user_id, limit)


@app.post("/posts/{post_id}/sync", dependencies=[Depends(_verify_api_key)])
def posts_sync(post_id: int) -> dict:
    import sqlite3 as _sqlite3
    from ml.feed_data_loader import load_feed_posts
    from ml.feed_embeddings import embed_posts

    db_path = _db_path_from_env()
    if db_path:
        _conn = _sqlite3.connect(db_path)
        try:
            posts = load_feed_posts(_conn)
        finally:
            _conn.close()
    else:
        from database import get_connection
        with get_connection() as conn:
            posts = load_feed_posts(conn)
    post_row = posts[posts["id"] == post_id]
    if post_row.empty:
        raise HTTPException(status_code=404, detail="Post not found")

    id_map_path = FEED_ARTIFACTS_DIR / "post_id_to_row.json"
    matrix_path = FEED_ARTIFACTS_DIR / "post_embeddings.npy"
    if not id_map_path.exists() or not matrix_path.exists():
        raise HTTPException(status_code=503, detail="Artifacts not found. Run initial training first.")

    # Serialize the read–modify–write critical section against concurrent syncs
    # so parallel POSTs don't corrupt the numpy matrix or the id map.
    with _sync_lock:
        id_to_row = {int(k): int(v) for k, v in json.loads(id_map_path.read_text()).items()}
        matrix = np.load(matrix_path)

        new_vec, _ = embed_posts(post_row)
        if post_id in id_to_row:
            matrix[id_to_row[post_id]] = new_vec[0]
        else:
            matrix = np.vstack([matrix, new_vec[0][np.newaxis, :]])
            id_to_row[post_id] = matrix.shape[0] - 1

        np.save(matrix_path, matrix)
        id_map_path.write_text(json.dumps({str(k): v for k, v in id_to_row.items()}))
    return {"status": "ok", "post_id": post_id, "action": "upserted"}


@app.delete("/posts/{post_id}/sync", dependencies=[Depends(_verify_api_key)])
def posts_sync_delete(post_id: int) -> dict:
    id_map_path = FEED_ARTIFACTS_DIR / "post_id_to_row.json"
    matrix_path = FEED_ARTIFACTS_DIR / "post_embeddings.npy"
    if not id_map_path.exists() or not matrix_path.exists():
        return {"status": "ok", "post_id": post_id, "action": "noop"}

    # Hold the lock across the whole read–modify–write. Otherwise a concurrent
    # POST could add the post between our miss check and our writeback.
    with _sync_lock:
        id_to_row = {int(k): int(v) for k, v in json.loads(id_map_path.read_text()).items()}
        if post_id not in id_to_row:
            return {"status": "ok", "post_id": post_id, "action": "noop"}

        matrix = np.load(matrix_path)
        row = id_to_row.pop(post_id)
        matrix = np.delete(matrix, row, axis=0)
        id_to_row = {pid: (idx if idx < row else idx - 1) for pid, idx in id_to_row.items()}
        np.save(matrix_path, matrix)
        id_map_path.write_text(json.dumps({str(k): v for k, v in id_to_row.items()}))
    return {"status": "ok", "post_id": post_id, "action": "deleted"}
