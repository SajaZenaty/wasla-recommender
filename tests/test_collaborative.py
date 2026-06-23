import pandas as pd

from src.ranking.collaborative import (
    build_interaction_matrix,
    compute_cf_scores,
    compute_item_similarity,
)


def test_empty_interactions_do_not_crash():
    interactions = pd.DataFrame(columns=["user_id", "post_id", "action", "timestamp"])
    posts = pd.DataFrame({"post_id": [1, 2, 3]})

    matrix, user_index, post_index = build_interaction_matrix(interactions, posts)
    similarity = compute_item_similarity(matrix)

    assert similarity.shape[0] == similarity.shape[1]

    idx_to_post_id = [None] * len(post_index)
    for pid, idx in post_index.items():
        idx_to_post_id[idx] = pid

    scores = compute_cf_scores(0, matrix, user_index, idx_to_post_id, similarity)
    assert scores == {}


def test_compute_cf_scores_with_sparse_similarity():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "post_id": 10, "action": "click"},
            {"user_id": 1, "post_id": 20, "action": "save"},
            {"user_id": 2, "post_id": 10, "action": "apply"},
        ]
    )
    posts = pd.DataFrame({"post_id": [10, 20, 30]})

    matrix, user_index, post_index = build_interaction_matrix(interactions, posts)
    similarity = compute_item_similarity(matrix)

    assert hasattr(similarity, "tocsr")

    idx_to_post_id = [None] * len(post_index)
    for pid, idx in post_index.items():
        idx_to_post_id[idx] = pid

    scores = compute_cf_scores(1, matrix, user_index, idx_to_post_id, similarity)

    assert isinstance(scores, dict)
    assert all(isinstance(v, float) for v in scores.values())
    assert 10 not in scores
    assert 20 not in scores
