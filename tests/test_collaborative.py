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

    scores = compute_cf_scores(0, matrix, user_index, post_index, similarity)
    assert scores == {}
