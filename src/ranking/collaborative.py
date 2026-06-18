import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

ACTION_WEIGHTS = {"click": 1, "save": 2, "apply": 4}

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
   
    if user_id not in user_index:
        return {}

    user_idx = user_index[user_id]
    user_vec = matrix[user_idx] 
    
    raw_scores = user_vec.dot(similarity).flatten()

    if hasattr(user_vec, 'indices'):
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


def compute_item_similarity(matrix):
    if matrix.shape[1] == 0:
        return np.array([[]])
    return cosine_similarity(matrix.T)