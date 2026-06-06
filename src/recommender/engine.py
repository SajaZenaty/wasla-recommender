import pandas as pd
import numpy as np

from src.features.embeddings import (
    build_post_embeddings,
    build_user_vectors,
    get_user_vector,
    is_zero_vector,
)
from recommender.retrieval import (
    build_faiss_index,
    dual_faiss_search,
    normalize_retrieval_scores,
)
from src.ranking.collaborative import (
    build_interaction_matrix,
    compute_item_similarity,
    compute_cf_scores,
)
from src.ranking.scoring import compute_hybrid_score, compute_similarity


MIN_CANDIDATES = 20
FAISS_TOP_K = 40


def bootstrap_system_data(users_df, posts_df, interactions_df):
    posts_df = posts_df.reset_index(drop=True)
    post_ids = posts_df["post_id"].tolist()

    post_embeddings = build_post_embeddings(posts_df)
    index, post_ids, post_id_to_idx = build_faiss_index(post_embeddings, post_ids)

    matrix, user_index, post_index = build_interaction_matrix(interactions_df, posts_df)
    similarity = compute_item_similarity(matrix)

    user_trust_map = users_df.set_index("user_id")["trust_score"].to_dict()
    posts_by_id = posts_df.set_index("post_id", drop=False)

    return {
        "post_embeddings": post_embeddings,
        "post_ids": post_ids,
        "post_id_to_idx": post_id_to_idx,
        "index": index,
        "matrix": matrix,
        "user_index": user_index,
        "post_index": post_index,
        "similarity": similarity,
        "user_trust_map": user_trust_map,
        "posts_by_id": posts_by_id,
    }


def _eligible_posts(user, posts_df):
    mask = (
        (posts_df["time_credits"] <= user["time_balance"]) &
        (posts_df["user_id"] != user["user_id"])
    )
    return posts_df[mask]


def _cold_start_candidates(user, eligible_df, existing_ids, needed):
    if needed <= 0:
        return {}

    extras = {}
    for _, post in eligible_df.iterrows():
        post_id = post["post_id"]
        if post_id in existing_ids:
            continue
        if compute_similarity(user, post) >= 1.0:
            extras[post_id] = 0.5
        if len(extras) >= needed:
            break

    return extras


def _retrieve_candidates(user, user_vectors, eligible_df, system_data):
    retrieval_raw = dual_faiss_search(
        system_data["index"],
        system_data["post_ids"],
        user_vectors["consumer"],
        user_vectors["provider"],
        top_k=FAISS_TOP_K,
    )

    eligible_ids = set(eligible_df["post_id"].tolist())
    retrieval_raw = {k: v for k, v in retrieval_raw.items() if k in eligible_ids}

    if is_zero_vector(user_vectors["consumer"]) and is_zero_vector(user_vectors["provider"]):
        retrieval_raw = {}

    if len(retrieval_raw) < MIN_CANDIDATES:
        needed = MIN_CANDIDATES - len(retrieval_raw)
        extras = _cold_start_candidates(user, eligible_df, set(retrieval_raw.keys()), needed)
        retrieval_raw.update(extras)

    return normalize_retrieval_scores(retrieval_raw)


def recommend(user, posts_df, interactions_df, system_data, top_k=10):
    user_id = user["user_id"]
    eligible_df = _eligible_posts(user, posts_df)

    if eligible_df.empty:
        return pd.DataFrame()

    cf_score_map = compute_cf_scores(
        user_id,
        system_data["matrix"],
        system_data["user_index"],
        system_data["post_index"],
        system_data["similarity"],
    )

    user_vectors = build_user_vectors(user, interactions_df, system_data)
    retrieval_scores = _retrieve_candidates(user, user_vectors, eligible_df, system_data)

    results = []
    posts_by_id = system_data["posts_by_id"]

    for post_id, retrieval_prior in retrieval_scores.items():
        if post_id not in posts_by_id.index:
            continue

        post = posts_by_id.loc[post_id]
        user_vec = get_user_vector(user_vectors, post)
        post_vec = system_data["post_embeddings"][system_data["post_id_to_idx"][post_id]]

        cf_s = cf_score_map.get(post_id, 0.0)
        author_trust = system_data["user_trust_map"].get(post["user_id"], 0.0)

        score, breakdown = compute_hybrid_score(
            user, post, user_vec, post_vec, cf_s, retrieval_prior, author_trust
        )

        row = post.to_dict()
        row.update(breakdown)
        results.append(row)

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values(by="final_score", ascending=False)
        .head(top_k)
    )
