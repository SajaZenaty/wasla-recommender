"""Runtime configuration for the HTTP service.

Values are read from environment variables (or a local .env file). Field names
map case-insensitively to env vars, e.g. ``express_internal_url`` <-
``EXPRESS_INTERNAL_URL``.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Auth + Express integration
    recommender_api_key: str | None = None
    express_internal_url: str | None = None
    express_timeout_ms: int = 5000

    # Startup / data source
    bootstrap_on_start: bool = True
    use_mock_data: bool = False
    mock_n_users: int = 50

    # Snapshot persistence
    index_snapshot_path: str = "/data/index_snapshot.pkl"

    # Scheduler
    enable_scheduler: bool = True
    nightly_rebuild_cron: str = "0 3 * * *"

    # Serving limits
    max_top_k: int = 50
    default_top_k: int = 10
    default_search_threshold: float = 0.4

    log_level: str = "INFO"


_settings: ServiceSettings | None = None


def get_settings() -> ServiceSettings:
    global _settings
    if _settings is None:
        _settings = ServiceSettings()
    return _settings
