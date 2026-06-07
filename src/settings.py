"""Central configuration for the recommender.

All tunable constants live here so they can be adjusted in one place and
overridden via environment variables without changing code or redeploying.
"""
import os


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)


FAISS_TOP_K = _env_int("FAISS_TOP_K", 40)
MIN_CANDIDATES = _env_int("MIN_CANDIDATES", 20)

ACTION_WEIGHTS = {
    "click": _env_float("ACTION_WEIGHT_CLICK", 1.0),
    "save": _env_float("ACTION_WEIGHT_SAVE", 2.0),
    "apply": _env_float("ACTION_WEIGHT_APPLY", 4.0),
}
LAMBDA_DECAY = _env_float("LAMBDA_DECAY", 0.05)

SCORING_WEIGHTS = {
    "semantic": _env_float("SCORING_WEIGHT_SEMANTIC", 0.35),
    "retrieval": _env_float("SCORING_WEIGHT_RETRIEVAL", 0.15),
    "cf": _env_float("SCORING_WEIGHT_CF", 0.15),
    "category": _env_float("SCORING_WEIGHT_CATEGORY", 0.10),
    "location": _env_float("SCORING_WEIGHT_LOCATION", 0.08),
    "time_fit": _env_float("SCORING_WEIGHT_TIME_FIT", 0.07),
    "freshness": _env_float("SCORING_WEIGHT_FRESHNESS", 0.05),
    "trust": _env_float("SCORING_WEIGHT_TRUST", 0.05),
}

TIME_BALANCE_THRESHOLDS = {
    "low": _env_float("TIME_BALANCE_LOW", 5),
    "high": _env_float("TIME_BALANCE_HIGH", 20),
    "bonus": _env_float("TIME_BALANCE_BONUS", 0.15),
    "penalty": _env_float("TIME_BALANCE_PENALTY", -0.05),
}
