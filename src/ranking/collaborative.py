import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.settings import ACTION_WEIGHTS
from src.utils.ids import lookup_in_mapping


def compute_item_similarity(matrix):
    """Item-item cosine similarity from the user x item interaction matrix.

    Returns a sparse (n_items x n_items) matrix so it can be multiplied with a
    sparse user row and still support ``.toarray()`` downstream.
    """
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        n_items = matrix.shape[1]
        return csr_matrix((n_items, n_items))
    return cosine_similarity(matrix.T, dense_output=False)


def build_interaction_matrix(interactions_df, posts_df):
    users = interactions_df["user_id"].unique()
    posts = posts_df["post_id"].unique()

    user_index = {u: i for i, u in enumerate(users)}
    post_index = {p: i for i, p in enumerate(posts)}

    rows = interactions_df["user_id"].map(user_index)
    cols = interactions_df["post_id"].map(post_index)
    data = interactions_df["action"].map(ACTION_WEIGHTS).fillna(1)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(posts)))

    return matrix, user_index, post_index


def compute_cf_scores(user_id, matrix, user_index, idx_to_post_id, similarity):
    user_idx = lookup_in_mapping(user_index, user_id)
    if user_idx is None:
        return {}

    user_vec = matrix[user_idx]

    raw_scores = user_vec.dot(similarity).toarray().ravel()

    if hasattr(user_vec, "indices"):
        seen = user_vec.indices
    else:
        seen = np.where(user_vec > 0)[0]

    raw_scores[seen] = -1

    pos_mask = raw_scores > 0
    pos_indices = np.where(pos_mask)[0]
    pos_scores = raw_scores[pos_indices]

    if pos_scores.size > 0:
        mn, mx = pos_scores.min(), pos_scores.max()
        if mx > mn:
            pos_scores = (pos_scores - mn) / (mx - mn)

    return {idx_to_post_id[i]: float(s) for i, s in zip(pos_indices, pos_scores)}
