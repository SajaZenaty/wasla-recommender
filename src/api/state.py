"""In-memory recommender state shared across requests.

Holds the three source-of-truth DataFrames (users, posts, interactions) and the
derived ``system_data`` (embeddings, FAISS index, CF matrix). All access is
guarded by a re-entrant lock.

Incremental pushes (new post / interaction / user) update the DataFrames and
mark the derived index dirty; the index is rebuilt lazily on the next
recommendation. This keeps push latency low while guaranteeing that pushed data
is reflected in the very next recommend call.
"""
import os
import pickle
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    ensure_interactions_schema,
    ensure_users_schema,
    preprocess_posts,
    preprocess_users,
)
from src.recommender.engine import bootstrap_system_data, recommend
from src.utils.ids import filter_by_id, id_variants, lookup_in_mapping

_BREAKDOWN_KEYS = [
    "semantic",
    "retrieval",
    "cf",
    "category_score",
    "location_score",
    "time_fit",
    "freshness",
    "trust",
    "balance_bias",
]


def _native(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _coerce_timestamps(df, column):
    if not df.empty and column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
        df[column] = df[column].dt.tz_localize(None)
    return df


class RecommenderState:
    def __init__(self):
        self._lock = threading.RLock()
        self.users_df = None
        self.posts_df = None
        self.interactions_df = None
        self.system_data = None
        self._dirty = False
        self.last_bootstrap_at = None
        self.model_loaded = False
        self.data_source: str | None = None

    @property
    def ready(self):
        return self.system_data is not None

    @property
    def post_count(self):
        return 0 if self.posts_df is None else len(self.posts_df)

    @property
    def user_count(self):
        return 0 if self.users_df is None else len(self.users_df)

    def can_serve_recommendations(self):
        return self.ready and self.user_count > 0

    def readiness_issue(self):
        if self.can_serve_recommendations():
            return None
        if not self.ready:
            return (
                "index_empty: set EXPRESS_INTERNAL_URL and POST /sync/bootstrap, "
                "or set USE_MOCK_DATA=true"
            )
        if self.user_count == 0:
            return (
                "users_missing: Express export must include users with matching "
                "user_id values, then POST /sync/bootstrap"
            )
        return None

    def status(self):
        with self._lock:
            return {
                "ready": self.ready,
                "can_serve_recommendations": self.can_serve_recommendations(),
                "model_loaded": self.model_loaded,
                "posts": self.post_count,
                "users": self.user_count,
                "interactions": 0
                if self.interactions_df is None
                else len(self.interactions_df),
                "last_bootstrap_at": self.last_bootstrap_at.isoformat()
                if self.last_bootstrap_at
                else None,
                "pending_rebuild": self._dirty,
                "data_source": self.data_source,
                "issue": self.readiness_issue(),
            }

    # ------------------------------------------------------------------
    # Bootstrap / rebuild
    # ------------------------------------------------------------------
    def set_data(self, users_df, posts_df, interactions_df):
        with self._lock:
            previous = (
                self.users_df,
                self.posts_df,
                self.interactions_df,
                self.system_data,
                self._dirty,
                self.last_bootstrap_at,
            )
            try:
                self.users_df = users_df.reset_index(drop=True)
                self.posts_df = posts_df.reset_index(drop=True)
                self.interactions_df = ensure_interactions_schema(interactions_df).reset_index(
                    drop=True
                )
                self._rebuild()
            except Exception:
                (
                    self.users_df,
                    self.posts_df,
                    self.interactions_df,
                    self.system_data,
                    self._dirty,
                    self.last_bootstrap_at,
                ) = previous
                raise

    def _rebuild(self):
        if self.posts_df is None or self.posts_df.empty:
            self.system_data = None
        else:
            users_df = ensure_users_schema(self.users_df)
            interactions_df = ensure_interactions_schema(self.interactions_df)
            self.system_data = bootstrap_system_data(
                users_df, self.posts_df, interactions_df
            )
        self._dirty = False
        self.last_bootstrap_at = datetime.now(timezone.utc)

    def _ensure_fresh(self):
        if self._dirty:
            self._rebuild()

    # ------------------------------------------------------------------
    # Incremental pushes
    # ------------------------------------------------------------------
    def upsert_post(self, post):
        with self._lock:
            row = _coerce_timestamps(pd.DataFrame([post]), "timestamp")
            row = preprocess_posts(row)
            if self.posts_df is None:
                self.posts_df = row
            else:
                variants = set(id_variants(post["post_id"]))
                kept = self.posts_df[~self.posts_df["post_id"].isin(variants)]
                self.posts_df = pd.concat([kept, row], ignore_index=True)
            self._dirty = True

    def add_interaction(self, interaction):
        with self._lock:
            row = _coerce_timestamps(pd.DataFrame([interaction]), "timestamp")
            if self.interactions_df is None:
                self.interactions_df = ensure_interactions_schema(row)
            else:
                self.interactions_df = ensure_interactions_schema(
                    pd.concat([self.interactions_df, row], ignore_index=True)
                )
            self._dirty = True

    def upsert_users(self, users):
        with self._lock:
            incoming = preprocess_users(pd.DataFrame(users))
            if self.users_df is None:
                self.users_df = incoming
            else:
                incoming_ids = {
                    variant
                    for user_id in incoming["user_id"].tolist()
                    for variant in id_variants(user_id)
                }
                kept = self.users_df[~self.users_df["user_id"].isin(incoming_ids)]
                self.users_df = pd.concat([kept, incoming], ignore_index=True)
            # User profile changes affect ranking but not the post index; still
            # mark dirty so cold-start vectors are recomputed on next request.
            self._dirty = True

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id, top_k):
        with self._lock:
            self._ensure_fresh()
            if not self.ready:
                return None, "index_not_ready"

            if self.users_df is None or self.users_df.empty:
                return None, "user_not_found"

            matches = filter_by_id(self.users_df, "user_id", user_id)
            if matches.empty:
                return None, "user_not_found"

            user = matches.iloc[0].to_dict()
            results = recommend(
                user,
                self.posts_df,
                self.interactions_df,
                self.system_data,
                top_k=top_k,
            )

            recs = []
            for _, row in results.iterrows():
                recs.append(
                    {
                        "post_id": _native(row["post_id"]),
                        "final_score": float(row["final_score"]),
                        "post_type": row.get("post_type"),
                        "breakdown": {
                            key: float(row[key])
                            for key in _BREAKDOWN_KEYS
                            if key in row
                        },
                    }
                )
            return recs, None

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------
    def save_snapshot(self, path):
        with self._lock:
            if self.posts_df is None:
                return False
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {
                "users_df": self.users_df,
                "posts_df": self.posts_df,
                "interactions_df": self.interactions_df,
                "saved_at": datetime.now(timezone.utc),
            }
            with open(path, "wb") as fh:
                pickle.dump(payload, fh)
            return True

    def load_snapshot(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.set_data(
            payload["users_df"],
            payload["posts_df"],
            payload.get("interactions_df"),
        )
        return True


state = RecommenderState()
