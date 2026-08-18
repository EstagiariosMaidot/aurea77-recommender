# Aurea77 Recommendation Service (v1 — LTR)

Hybrid **Learning-to-Rank** recommender for Aurea77 events and social posts.
Event ranking combines event-description embeddings with each athlete's
`sports_practiced`, geo distance, seasonality, popularity, ratings and history.
It does not depend on `categories` or `category_event`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///.../database.sqlite` | Database connection. SQLite for dev, MySQL for production (`mysql://user:pass@host:3306/dbname`) |
| `RECOMMENDER_API_KEY` | required | API key for authenticating requests |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON logs (production) |
| `LOG_LEVEL` | `INFO` | Logging level |

When `DATABASE_URL` is unset, the service also accepts Laravel's
`DB_CONNECTION`, `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME`, and
`DB_PASSWORD` variables. This avoids storing a second copy of database
credentials for local API integration.

## Modes

### 1. Serve

```bash
# Local (SQLite)
uvicorn main:app --port 5000

# Production (MySQL)
DATABASE_URL=mysql://user:pass@host:3306/aurea77 \
RECOMMENDER_API_KEY=your-secret-key \
LOG_FORMAT=json \
uvicorn main:app --port 5000
```

### 2. Train

```bash
# Local
python -m train --db-path ../aurea77-api/database/database.sqlite

# Production
DATABASE_URL=mysql://user:pass@host:3306/aurea77 python -m train
```

Options: `--n-estimators 300`, `--cutoff-percentile 0.80`, `--neg-ratio 5`, `--seed 42`.

Outputs: `artifacts/{model.pkl, event_embeddings.npy, event_id_to_row.json, metadata.json}`.
Uses atomic swap — the service is never interrupted during training.

### 3. Evaluate

```bash
python -m evaluate --db-path ../aurea77-api/database/database.sqlite
```

Compares LTR against Random / Popularity / sports-profile baselines on a temporal split.
Prints feature importances and per-segment Hit@k. Exits non-zero if the quality
gate fails.

### 4. Retrain (automated pipeline)

```bash
# Local
python -m retrain --db-path ../aurea77-api/database/database.sqlite

# Production
DATABASE_URL=mysql://user:pass@host:3306/aurea77 python -m retrain

# Test without promoting
python -m retrain --db-path ... --dry-run
```

Pipeline: **train → evaluate → promote (or abort)**.
Only promotes the new model if the quality gate passes. The previous model is
backed up to `artifacts_backup/`. Failed models are saved to
`artifacts_failed_<timestamp>/` for inspection.

Cron example (weekly, Sundays at 03:00):
```cron
0 3 * * 0  cd /opt/aurea77-recommendation && .venv/bin/python -m retrain >> /var/log/aurea77-retrain.log 2>&1
```

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | No | Service status, uptime, model version, embeddings count |
| `GET` | `/metrics` | API key | Operational metrics (latency, errors, recommendation stats) |
| `GET` | `/recommend/{user_id}?limit=10` | API key | Top-N recommendations for a user |
| `GET` | `/events/embedded-ids` | API key | List event IDs that have embeddings |
| `POST` | `/events/{event_id}/sync` | API key | Compute/update embedding for an event |
| `DELETE` | `/events/{event_id}/sync` | API key | Remove an event embedding |
| `GET` | `/posts/feed/{user_id}?limit=10` | API key | Ordered mixed post feed |
| `POST` | `/posts/feed?limit=10` | API key + bearer JWT | Ordered feed for the authenticated user |

Authentication: send `X-API-Key: your-key` header.

### Data conventions

- `event_registrations.created_at` is the canonical interaction timestamp for
  temporal training, evaluation and candidate exclusion. `registered_at` is
  not read because historic exports leave it incomplete.
- Training and offline evaluation are point-in-time: user history, popularity,
  reviews and candidate availability are calculated strictly from data known
  before each registration timestamp.
- `finished` registrations are normalised to `completed` at ingestion, along
  with the canonical `registered` and `planned` statuses.
- Event content is embedded from its name and description. User sports are
  embedded from `athlete_profiles.sports_practiced`; no category tables are
  required for event recommendations.

### Feed ranking

`GET /posts/feed/{user_id}` and `POST /posts/feed` return the ordered mixed
feed. Candidate privacy, friendships, blocks and mutes remain hard filters.
When feed-model artifacts have not been trained, the endpoints return the same
eligible candidates in reverse chronological order instead of failing.

## Event Sync (real-time updates)

When an event is created or edited in the Laravel API, the `EventObserver`
automatically dispatches a queue job that calls `POST /events/{id}/sync`.
When an event is deleted, the observer calls `DELETE /events/{id}/sync`
synchronously.

This means new/edited events become available in recommendations within
seconds, without waiting for a retrain.

## Monitoring

- `GET /health` — uptime, model version, when it was trained, number of embeddings
- `GET /metrics` — counters (request count, errors, syncs), histograms (latency, result count, scores)
- Structured JSON logging with `LOG_FORMAT=json`

## Laravel API Integration

Files added to `aurea77-api`:

- `app/Http/Controllers/Api/V1/RecommendationController.php` — `GET /api/v1/me/recommendations`
- `app/Observers/EventObserver.php` — auto-sync on event create/edit/delete
- `app/Jobs/SyncRecommenderEmbeddingJob.php` — async job with 3 retries
- `app/Console/Commands/RecommenderCatchUpCommand.php` — `php artisan recommender:catch-up`
- `config/services.php` — `recommender.url` and `recommender.api_key`

## Tests

```bash
pytest -v
```

The suite covers features, embeddings, data loading, labels, splits, metrics,
artifacts, recommender inference, sync endpoints, feed ranking and monitoring.
Tests use an in-memory SQLite fixture — they don't require the Laravel database.
