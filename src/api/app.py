"""FastAPI service exposing the Wasla recommender.

Endpoints:
    GET  /health            liveness
    GET  /ready             readiness + data status
    POST /recommend         ranked posts for a user
    POST /sync/bootstrap     full rebuild (inline payload or pull from Express)
    POST /sync/post          upsert a single post
    POST /sync/interaction   record an interaction
    POST /sync/users         batch upsert user profiles

Write/sync endpoints require the ``X-Internal-Token`` header when an API key is
configured.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from src.api.config import get_settings
from src.api.schemas import (
    BootstrapRequest,
    InteractionIn,
    PostIn,
    RecommendRequest,
    RecommendResponse,
    StatusResponse,
    UsersSyncRequest,
)
from src.api.state import state
from src.data.express_loader import frames_from_payload, load_from_express

logger = logging.getLogger("wasla.api")


def _api_error(status_code, error, message, retry_after=None):
    detail = {"error": error, "message": message}
    if retry_after is not None:
        detail["retry_after"] = retry_after
    return HTTPException(status_code=status_code, detail=detail)


def preload_model():
    """Load the embedding model once so the first request is not slow."""
    try:
        from src.features.embeddings import EmbeddingModel

        EmbeddingModel.get_model()
        state.model_loaded = True
        logger.info("Embedding model loaded")
    except Exception as exc:  # noqa: BLE001 - startup must not hard-crash here
        logger.warning("Could not preload embedding model: %s", exc)


def do_full_rebuild():
    """Pull a fresh snapshot from Express and rebuild, then persist to disk."""
    settings = get_settings()
    if not settings.express_internal_url:
        logger.warning("Nightly rebuild skipped: EXPRESS_INTERNAL_URL not set")
        return False
    try:
        users_df, posts_df, interactions_df = load_from_express(
            settings.express_internal_url,
            api_key=settings.recommender_api_key,
            timeout_ms=settings.express_timeout_ms,
        )
        state.set_data(users_df, posts_df, interactions_df)
        state.save_snapshot(settings.index_snapshot_path)
        logger.info("Full rebuild from Express complete (%d posts)", state.post_count)
        return True
    except Exception as exc:  # noqa: BLE001 - keep serving last good index
        logger.error("Full rebuild failed, keeping previous index: %s", exc)
        return False


def load_initial_data():
    settings = get_settings()
    if not settings.bootstrap_on_start:
        logger.info("Bootstrap on start disabled")
        return

    if state.load_snapshot(settings.index_snapshot_path):
        logger.info("Loaded snapshot from %s", settings.index_snapshot_path)
        return

    if settings.use_mock_data:
        from src.data.loader import load_mock_data

        users_df, posts_df, interactions_df = load_mock_data(
            n_users=settings.mock_n_users, seed=42
        )
        state.set_data(users_df, posts_df, interactions_df)
        logger.info("Bootstrapped from mock data (%d posts)", state.post_count)
        return

    if settings.express_internal_url:
        do_full_rebuild()
        return

    logger.warning("No data source configured; service starts in not-ready state")


def _start_scheduler():
    settings = get_settings()
    if not settings.enable_scheduler:
        return None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            do_full_rebuild,
            CronTrigger.from_crontab(settings.nightly_rebuild_cron),
            id="nightly_rebuild",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Scheduler started: cron='%s'", settings.nightly_rebuild_cron)
        return scheduler
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scheduler not started: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    preload_model()
    load_initial_data()
    scheduler = _start_scheduler()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)


app = FastAPI(title="Wasla Recommender", version="1.0.0", lifespan=lifespan)


def require_token(x_internal_token: str | None = Header(default=None)):
    settings = get_settings()
    if not settings.recommender_api_key:
        return  # auth disabled (dev mode)
    if x_internal_token != settings.recommender_api_key:
        raise _api_error(401, "unauthorized", "Invalid or missing X-Internal-Token")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.model_loaded}


@app.get("/ready", response_model=StatusResponse)
def ready():
    return state.status()


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest, _=Depends(require_token)):
    settings = get_settings()
    top_k = req.top_k or settings.default_top_k
    top_k = min(top_k, settings.max_top_k)

    recs, error = state.recommend_for_user(req.user_id, top_k)
    if error == "index_not_ready":
        raise _api_error(503, error, "Recommender index is not ready", retry_after=30)
    if error == "user_not_found":
        raise _api_error(404, error, f"Unknown user_id: {req.user_id}")

    return {"user_id": req.user_id, "count": len(recs), "recommendations": recs}


@app.post("/sync/bootstrap")
def sync_bootstrap(req: BootstrapRequest | None = None, _=Depends(require_token)):
    settings = get_settings()
    has_inline = req is not None and req.posts is not None
    if has_inline:
        users_df, posts_df, interactions_df = frames_from_payload(req.model_dump())
        state.set_data(users_df, posts_df, interactions_df)
        state.save_snapshot(settings.index_snapshot_path)
        return {"status": "ok", "source": "payload", "posts": state.post_count}

    ok = do_full_rebuild()
    if not ok:
        raise _api_error(503, "rebuild_failed", "Could not pull data from Express")
    return {"status": "ok", "source": "express", "posts": state.post_count}


@app.post("/sync/post")
def sync_post(post: PostIn, _=Depends(require_token)):
    state.upsert_post(post.model_dump())
    return {"status": "ok", "post_id": post.post_id}


@app.post("/sync/interaction")
def sync_interaction(interaction: InteractionIn, _=Depends(require_token)):
    state.add_interaction(interaction.model_dump())
    return {"status": "ok"}


@app.post("/sync/users")
def sync_users(req: UsersSyncRequest, _=Depends(require_token)):
    state.upsert_users([u.model_dump() for u in req.users])
    return {"status": "ok", "users": state.user_count}
