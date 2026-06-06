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

def compute_cf_scores(user_id, matrix, user_index, post_index, similarity):
    if user_id not in user_index:
        return {}

    user_idx = user_index[user_id]
    user_vec = matrix[user_idx]  
    
    raw_scores = user_vec.dot(similarity).toarray().flatten()

    seen = user_vec.indices 
    raw_scores[seen] = -1

    cf_score_map = {}
    for pid, idx in post_index.items():
        if raw_scores[idx] > 0:
            cf_score_map[pid] = float(raw_scores[idx])

    if cf_score_map:
        vals = np.array(list(cf_score_map.values()))
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            for k in cf_score_map:
                cf_score_map[k] = (cf_score_map[k] - mn) / (mx - mn)

    return cf_score_map