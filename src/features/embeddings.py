import numpy as np
import pandas as pd
from datetime import datetime
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from src.settings import (
    ACTION_WEIGHTS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    LAMBDA_DECAY,
)
from src.data.preprocessing import is_offer_post
from src.utils.ids import filter_by_id, lookup_in_mapping


class EmbeddingModel:
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._instance


def is_zero_vector(vec, eps=1e-8):
    if vec is None:
        return True
    return float(np.linalg.norm(vec)) < eps


def get_user_vector(user_vectors, post):
    """Pick the user representation relevant to a given post."""
    if is_offer_post(post.get("post_type")):
        return user_vectors["consumer"]
    return user_vectors["provider"]


def embed_batch(texts, batch_size=EMBEDDING_BATCH_SIZE):
    model = EmbeddingModel.get_model()
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)


@lru_cache(maxsize=128)
def _cached_weighted_embedding(items_tuple):
    model = EmbeddingModel.get_model()
    items = list(items_tuple)

    if not items:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

    item_vecs = model.encode(items, normalize_embeddings=True)
    avg_vec = np.mean(item_vecs, axis=0)
    norm = np.linalg.norm(avg_vec)
    return (avg_vec / norm).astype(np.float32) if norm > 0 else avg_vec


def build_weighted_embedding(items):
    return _cached_weighted_embedding(tuple(items))


def build_post_embeddings(posts_df):
    texts = [
        f"عنوان: {row['title_clean']} وصف: {row['desc_clean']} تصنيف: {row['category']}"
        for _, row in posts_df.iterrows()
    ]
    return embed_batch(texts).astype(np.float32)


def build_user_interaction_embedding(user_id, interactions_df, post_embeddings, post_id_to_idx, posts_df=None, target_post_type=None):
    user_interactions = filter_by_id(interactions_df, "user_id", user_id)
    if user_interactions.empty:
        return None

    # منطق الفلترة الجديد: إذا تم تمرير بيانات المنشورات، نقوم بفلترة التفاعلات حسب النوع
    if posts_df is not None and target_post_type is not None:
        # ربط التفاعلات مع نوع المنشور
        merged_df = user_interactions.merge(posts_df[['post_id', 'post_type']], on='post_id', how='left')
        is_offer_target = (target_post_type == "offer")
        user_interactions = merged_df[merged_df['post_type'].apply(is_offer_post) == is_offer_target]

    if user_interactions.empty:
        return None

    matched_rows = []
    post_indices = []
    for _, row in user_interactions.iterrows():
        idx = lookup_in_mapping(post_id_to_idx, row["post_id"])
        if idx is not None:
            matched_rows.append(row)
            post_indices.append(idx)

    if not post_indices:
        return None

    vecs = post_embeddings[post_indices]
    matched = pd.DataFrame(matched_rows)

    now = datetime.now()
    days_diff = (now - pd.to_datetime(matched["timestamp"])).dt.days
    decay = np.exp(-LAMBDA_DECAY * np.maximum(0, days_diff))

    weights = matched["action"].map(ACTION_WEIGHTS).fillna(1) * decay

    avg = np.average(vecs, axis=0, weights=weights)
    norm = np.linalg.norm(avg)

    return (avg / norm).astype(np.float32) if norm > 0 else None


def build_user_vectors(user, interactions_df, system_data):
    vec_needs = build_weighted_embedding(user.get("needs", []))
    vec_offers = build_weighted_embedding(user.get("skills", []))

    # بناء متجه التفاعل الخاص بالاستهلاك (تفاعل مع العروض)
    int_vec_consumer = build_user_interaction_embedding(
        user["user_id"], interactions_df,
        system_data["post_embeddings"], system_data["post_id_to_idx"],
        system_data["posts_by_id"], "offer"
    )

    # بناء متجه التفاعل الخاص بالتقديم (تفاعل مع الطلبات)
    int_vec_provider = build_user_interaction_embedding(
        user["user_id"], interactions_df,
        system_data["post_embeddings"], system_data["post_id_to_idx"],
        system_data["posts_by_id"], "request"
    )

    def combine(base_vec, interaction_vec):
        if interaction_vec is not None:
            # دمج بنسبة 60/40
            v = 0.6 * base_vec + 0.4 * interaction_vec
            norm = np.linalg.norm(v)
            return (v / norm).astype(np.float32) if norm > 0 else base_vec
        return base_vec

    return {
        "consumer": combine(vec_needs, int_vec_consumer),
        "provider": combine(vec_offers, int_vec_provider),
    }
