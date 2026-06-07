# Wasla Recommender

A hybrid recommendation service for a time-banking / skill-sharing platform
(Arabic-first). It ranks posts ("offers" and "requests") for each user by
combining semantic embeddings, collaborative filtering, and contextual signals
(category, location, time balance, freshness, author trust).

The engine runs as a standalone HTTP microservice (FastAPI) that a Node.js
Express backend calls over HTTP. Express stays the system of record; the
recommender keeps an in-memory index that is synced from Express.

## Architecture

```
Client ──> Express API ──> Python Recommender (FastAPI)
              │                    │
              │  push events       │  nightly pull + rebuild
              └────────────────────┘
                       │
                  /data snapshot (fast restarts)
```

- Real-time: Express pushes new posts / interactions / user updates.
- Nightly: the recommender pulls a full snapshot from Express and rebuilds.
- Restarts: the latest snapshot on the `/data` volume is loaded first.

## Project structure

```
src/
  api/            FastAPI app, config, request state, schemas
  data/           mock loader, Express loader, preprocessing
  features/       embedding model + user/post vectors
  ranking/        scoring + collaborative filtering
  recommender/    retrieval (FAISS) + the recommend() pipeline
  evaluation/     offline metrics (local / CI only)
  settings.py     single source of truth for tunable constants
scripts/          mock data generators
tests/            pytest suite
main.py           offline demo over mock data
```

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Offline demo over mock data
python main.py

# Run the API against mock data (no Express needed)
USE_MOCK_DATA=true ENABLE_SCHEDULER=false \
  uvicorn src.api.app:app --reload --port 8000
```

## Docker

```bash
cp .env.example .env   # then edit values
docker compose up --build
```

The image pre-downloads the embedding model at build time, so the container
needs no network at startup. Snapshots persist on the `recommender_data`
volume mounted at `/data`.

## Configuration

All settings are environment variables (see `.env.example`). Key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RECOMMENDER_API_KEY` | empty | Shared secret for `X-Internal-Token`. Empty disables auth (dev only). |
| `EXPRESS_INTERNAL_URL` | empty | Base URL of the Express export endpoint. |
| `USE_MOCK_DATA` | false | Bootstrap from generated mock data. |
| `BOOTSTRAP_ON_START` | true | Load snapshot / mock / Express on startup. |
| `INDEX_SNAPSHOT_PATH` | `/data/index_snapshot.pkl` | Snapshot location. |
| `ENABLE_SCHEDULER` | true | Run the nightly rebuild. |
| `NIGHTLY_REBUILD_CRON` | `0 3 * * *` | Cron for the nightly rebuild. |
| `MAX_TOP_K` / `DEFAULT_TOP_K` | 50 / 10 | Serving limits. |

Scoring weights can be overridden per signal, e.g. `SCORING_WEIGHT_SEMANTIC=0.4`,
without code changes. See `src/settings.py` for the full list.

## API reference

Write/sync endpoints require the `X-Internal-Token` header when
`RECOMMENDER_API_KEY` is set.

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET | `/health` | – | Liveness + model-loaded flag. |
| GET | `/ready` | – | Readiness + data counts + last rebuild. |
| POST | `/recommend` | `{user_id, top_k?}` | Ranked post IDs with score breakdown. |
| POST | `/sync/bootstrap` | snapshot or empty | Full rebuild from payload or Express pull. |
| POST | `/sync/post` | post object | Upsert a single post. |
| POST | `/sync/interaction` | interaction object | Record a click / save / apply. |
| POST | `/sync/users` | `{users: [...]}` | Batch upsert user profiles. |

### `/recommend` response

```json
{
  "user_id": 0,
  "count": 2,
  "recommendations": [
    {
      "post_id": 42,
      "final_score": 0.83,
      "post_type": "عرض",
      "breakdown": {"semantic": 0.71, "cf": 0.4, "category_score": 1.0, "...": 0.0}
    }
  ]
}
```

Pushes update the in-memory data and mark the index dirty; it is rebuilt lazily
on the next `/recommend`, so pushed data is reflected immediately while push
latency stays low.

## Data schema

See [docs/express-integration.md](docs/express-integration.md) for the exact
field contract Express must provide, the export endpoint, the feed proxy, and
the event push hooks.

## Testing

```bash
pytest          # unit + API tests
ruff check .    # lint
```

CI runs both on every pull request (see `.github/workflows/ci.yml`).

## Pre-release smoke test

1. `docker compose up` then `GET /ready` returns `ready: true`.
2. `POST /recommend` returns ordered post IDs for a known user.
3. Push a post via `POST /sync/post`, then `POST /recommend` includes it.
4. Restart the container; it loads the snapshot without contacting Express.
5. Trigger `POST /sync/bootstrap` and confirm a full rebuild completes with no
   downtime (the previous index keeps serving on failure).
