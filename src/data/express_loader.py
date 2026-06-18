"""Adapter that turns Express backend data into recommender DataFrames.

The Express side is expected to expose an internal, token-protected endpoint
that returns a full snapshot of users, posts and interactions. The same
payload shape is also accepted directly by the ``/sync/bootstrap`` endpoint.
"""
import httpx
import pandas as pd

from src.data.preprocessing import preprocess_posts, preprocess_users
from src.utils.validators import validate_posts_data, validate_users_data


def _coerce_timestamps(df, column):
    if not df.empty and column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
        df[column] = df[column].dt.tz_localize(None)
    return df


def frames_from_payload(payload):
    """Build preprocessed (users, posts, interactions) frames from a snapshot."""
    users_df = pd.DataFrame(payload.get("users", []))
    posts_df = pd.DataFrame(payload.get("posts", []))
    interactions_df = pd.DataFrame(payload.get("interactions", []))

    posts_df = _coerce_timestamps(posts_df, "timestamp")
    interactions_df = _coerce_timestamps(interactions_df, "timestamp")

    validate_users_data(users_df)
    validate_posts_data(posts_df)

    return (
        preprocess_users(users_df),
        preprocess_posts(posts_df),
        interactions_df,
    )


def fetch_snapshot(base_url, api_key=None, timeout_ms=5000):
    """Pull a full data snapshot from the Express internal export endpoint."""
    url = base_url.rstrip("/") + "/internal/recommender-export"
    headers = {}
    if api_key:
        headers["X-Internal-Token"] = api_key

    response = httpx.get(url, headers=headers, timeout=timeout_ms / 1000)
    response.raise_for_status()
    return response.json()


def load_from_express(base_url, api_key=None, timeout_ms=5000):
    payload = fetch_snapshot(base_url, api_key=api_key, timeout_ms=timeout_ms)
    return frames_from_payload(payload)
