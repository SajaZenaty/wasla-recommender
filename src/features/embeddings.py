import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import pandas as pd 

class EmbeddingModel:
    _instance = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return cls._instance

ACTION_WEIGHTS = {"click": 1, "save": 2, "apply": 4}
LAMBDA_DECAY = 0.05

def embed_batch(texts, batch_size=32):
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

def build_user_interaction_embedding(user_id, interactions_df, post_embeddings, post_id_to_idx):
    user_interactions = interactions_df[interactions_df["user_id"] == user_id]
    if user_interactions.empty:
        return None

    post_indices = [post_id_to_idx[pid] for pid in user_interactions["post_id"] if pid in post_id_to_idx]
    if not post_indices:
        return None
    
    vecs = post_embeddings[post_indices]
    
    now = datetime.now()
    days_diff = (now - pd.to_datetime(user_interactions["timestamp"])).dt.days
    decay = np.exp(-LAMBDA_DECAY * np.maximum(0, days_diff))
    
    weights = user_interactions["action"].map(ACTION_WEIGHTS).fillna(1) * decay
    
    avg = np.average(vecs, axis=0, weights=weights)
    norm = np.linalg.norm(avg)
    
    return (avg / norm).astype(np.float32) if norm > 0 else None

def build_user_vectors(user, interactions_df, system_data):
    vec_needs = build_weighted_embedding(user.get("needs", []))
    vec_offers = build_weighted_embedding(user.get("skills", []))

    interaction_vec = build_user_interaction_embedding(
        user["user_id"], interactions_df, 
        system_data["post_embeddings"], system_data["post_id_to_idx"]
    )

    def combine(base_vec):
        if interaction_vec is not None:
            v = 0.6 * base_vec + 0.4 * interaction_vec
            norm = np.linalg.norm(v)
            return (v / norm).astype(np.float32) if norm > 0 else base_vec
        return base_vec

    return {
        "consumer": combine(vec_needs), 
        "provider": combine(vec_offers), 
    }

def is_zero_vector(vec):
    if vec is None: return True
    return np.all(vec == 0)

def get_user_vector(user_vectors, post):
    if post["post_type"] == "طلب":
        return user_vectors["consumer"]
    return user_vectors["provider"]