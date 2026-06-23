"""FastAPI service exposing the Wasla recommender.

Endpoints:
    GET  /                  service info + endpoint links
    GET  /health            liveness
    GET  /ready             readiness + data status
    POST /recommend         ranked posts for a user
    POST /search            semantic search over posts
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
    SearchRequest,
    SearchResponse,
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


def _effective_express_url(settings):
    url = (settings.express_internal_url or "").strip()
    return url or None


def do_full_rebuild():
    """Pull a fresh snapshot from Express and rebuild, then persist to disk."""
    settings = get_settings()
    express_url = _effective_express_url(settings)
    if not express_url:
        logger.warning("Nightly rebuild skipped: EXPRESS_INTERNAL_URL not set")
        return False
    try:
        users_df, posts_df, interactions_df = load_from_express(
            express_url,
            api_key=settings.recommender_api_key,
            timeout_ms=settings.express_timeout_ms,
        )
        state.set_data(users_df, posts_df, interactions_df)
        if not _bootstrap_is_usable():
            logger.error(
                "Express export loaded but index is unusable (users=%d, posts=%d)",
                state.user_count,
                state.post_count,
            )
            return False
        state.save_snapshot(settings.index_snapshot_path)
        logger.info("Full rebuild from Express complete (%d posts)", state.post_count)
        return True
    except Exception as exc:  # noqa: BLE001 - keep serving last good index
        logger.error("Full rebuild failed, keeping previous index: %s", exc)
        return False


def _bootstrap_is_usable():
    """True when the loaded index can answer /recommend for known users."""
    if state.post_count > 0 and state.user_count == 0:
        return False
    return state.ready


def _log_bootstrap_summary(source: str):
    state.data_source = source
    summary = state.status()
    logger.info(
        "Bootstrap complete (%s): users=%d posts=%d interactions=%d can_serve=%s",
        source,
        summary["users"],
        summary["posts"],
        summary["interactions"],
        summary["can_serve_recommendations"],
    )
    if summary["ready"] and summary["users"] == 0:
        logger.error(
            "Index has posts but zero users — every /recommend will return 404 "
            "(user_not_found). Ensure Express export includes users with matching "
            "user_id values, then POST /sync/bootstrap."
        )


def load_initial_data():
    settings = get_settings()
    express_url = _effective_express_url(settings)
    logger.info(
        "Bootstrap config: bootstrap_on_start=%s use_mock_data=%s express_url=%s mock_n_users=%d",
        settings.bootstrap_on_start,
        settings.use_mock_data,
        "set" if express_url else "unset",
        settings.mock_n_users,
    )

    if not settings.bootstrap_on_start:
        logger.info("Bootstrap on start disabled (BOOTSTRAP_ON_START=false)")
        return

    if state.load_snapshot(settings.index_snapshot_path):
        if _bootstrap_is_usable():
            logger.info("Loaded snapshot from %s", settings.index_snapshot_path)
            _log_bootstrap_summary("snapshot")
            return
        logger.warning(
            "Snapshot at %s is unusable (users=%d, posts=%d); re-bootstrapping",
            settings.index_snapshot_path,
            state.user_count,
            state.post_count,
        )

    if express_url:
        if do_full_rebuild():
            _log_bootstrap_summary("express")
            return
        logger.warning("Express bootstrap failed or empty; trying fallback data source")

    if settings.use_mock_data and _bootstrap_from_mock():
        return

    if not express_url and _bootstrap_from_mock():
        return

    # Last resort: Express misconfigured or mock with too many users (OOM on free tier).
    logger.warning("Trying emergency mock bootstrap with 10 users")
    if _bootstrap_from_mock_n(10):
        return

    if not state.ready:
        logger.error(
            "Startup finished with empty index (ready=false). "
            "On Hugging Face: remove EXPRESS_INTERNAL_URL, set USE_MOCK_DATA=true, "
            "MOCK_N_USERS=10, then Factory rebuild."
        )


def _bootstrap_from_mock_n(n_users: int) -> bool:
    from src.data.loader import load_mock_data

    try:
        users_df, posts_df, interactions_df = load_mock_data(n_users=n_users, seed=42)
        state.set_data(users_df, posts_df, interactions_df)
    except Exception as exc:  # noqa: BLE001
        logger.error("Mock bootstrap failed (n_users=%d): %s", n_users, exc)
        return False
    if not _bootstrap_is_usable():
        logger.error("Mock bootstrap finished but index is still not ready (n_users=%d)", n_users)
        return False
    _log_bootstrap_summary("mock")
    return True


def _bootstrap_from_mock():
    settings = get_settings()
    return _bootstrap_from_mock_n(settings.mock_n_users)


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
    expected = settings.recommender_api_key.strip()
    provided = (x_internal_token or "").strip()
    if provided != expected:
        raise _api_error(401, "unauthorized", "Invalid or missing X-Internal-Token")


@app.get("/")
def root():
    summary = state.status()
    return {
        "service": "Wasla Recommender",
        "version": "1.0.0",
        "ready": summary["ready"],
        "can_serve_recommendations": summary["can_serve_recommendations"],
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs",
            "recommend": "POST /recommend",
            "search": "POST /search",
        },
    }


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
        summary = state.status()
        logger.warning(
            "user_not_found user_id=%r users_indexed=%d posts=%d — "
            "sync user via POST /sync/users or rebuild via POST /sync/bootstrap",
            req.user_id,
            summary["users"],
            summary["posts"],
        )
        raise _api_error(404, error, f"Unknown user_id: {req.user_id}")

    return {"user_id": req.user_id, "count": len(recs), "recommendations": recs}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest, _=Depends(require_token)):
    settings = get_settings()
    top_k = min(req.top_k or settings.default_top_k, settings.max_top_k)
    threshold = (
        req.threshold
        if req.threshold is not None
        else settings.default_search_threshold
    )

    results, error = state.search_for_query(req.query, top_k, threshold)
    if error == "index_not_ready":
        raise _api_error(
            503, error, "Search index is not ready", retry_after=30
        )

    return {"query": req.query, "count": len(results), "results": results}


def _has_inline_payload(req: BootstrapRequest | None) -> bool:
    if req is None:
        return False
    return any(
        getattr(req, field) is not None for field in ("users", "posts", "interactions")
    )


@app.post("/sync/bootstrap")
def sync_bootstrap(req: BootstrapRequest | None = None, _=Depends(require_token)):
    settings = get_settings()
    if _has_inline_payload(req):
        try:
            users_df, posts_df, interactions_df = frames_from_payload(req.model_dump())
            state.set_data(users_df, posts_df, interactions_df)
        except ValueError as exc:
            raise _api_error(400, "invalid_payload", str(exc)) from exc
        state.save_snapshot(settings.index_snapshot_path)
        return {"status": "ok", "source": "payload", "posts": state.post_count}

    ok = do_full_rebuild()
    if not ok:
        raise _api_error(503, "rebuild_failed", "Could not pull data from Express")
    return {"status": "ok", "source": "express", "posts": state.post_count}


@app.post("/sync/post")
def sync_post(post: PostIn, _=Depends(require_token)):
    try:
        state.upsert_post(post.model_dump())
    except ValueError as exc:
        raise _api_error(400, "invalid_payload", str(exc)) from exc
    return {"status": "ok", "post_id": post.post_id}


@app.post("/sync/interaction")
def sync_interaction(interaction: InteractionIn, _=Depends(require_token)):
    try:
        state.add_interaction(interaction.model_dump())
    except ValueError as exc:
        raise _api_error(400, "invalid_payload", str(exc)) from exc
    return {"status": "ok"}


@app.post("/sync/users")
def sync_users(req: UsersSyncRequest, _=Depends(require_token)):
    try:
        state.upsert_users([u.model_dump() for u in req.users])
    except ValueError as exc:
        raise _api_error(400, "invalid_payload", str(exc)) from exc
    return {"status": "ok", "users": state.user_count}
