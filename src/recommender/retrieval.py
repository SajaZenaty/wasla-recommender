import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class NumpyIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings.astype("float32")

    def search(self, query, top_k):
        scores = self.embeddings @ query.reshape(-1).astype("float32")
        k = min(top_k, len(scores))
        if k == 0:
            return np.array([[]]), np.array([[]])

        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return scores[top_indices].reshape(1, -1), top_indices.reshape(1, -1)


def build_faiss_index(post_embeddings, post_ids):
    post_ids = list(post_ids)
    post_id_to_idx = {pid: i for i, pid in enumerate(post_ids)}
    embeddings = post_embeddings.astype("float32")

    if FAISS_AVAILABLE:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        index = NumpyIndex(embeddings)

    return index, post_ids, post_id_to_idx


def faiss_search(index, post_ids, query_vec, top_k=40):
    if np.linalg.norm(query_vec) < 1e-8:
        return {}

    distances, indices = index.search(
        query_vec.reshape(1, -1).astype("float32"),
        min(top_k, len(post_ids)),
    )

    scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        post_id = post_ids[int(idx)]
        scores[post_id] = max(scores.get(post_id, 0.0), float(dist))

    return scores


def dual_faiss_search(index, post_ids, needs_vec, offers_vec, top_k=40):
    needs_scores = faiss_search(index, post_ids, needs_vec, top_k)
    offers_scores = faiss_search(index, post_ids, offers_vec, top_k)

    merged = {}
    for post_id, score in needs_scores.items():
        merged[post_id] = max(merged.get(post_id, 0.0), score)
    for post_id, score in offers_scores.items():
        merged[post_id] = max(merged.get(post_id, 0.0), score)

    return merged


def normalize_retrieval_scores(retrieval_scores):
    if not retrieval_scores:
        return {}

    vals = list(retrieval_scores.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {k: 1.0 for k in retrieval_scores}

    return {k: (v - mn) / (mx - mn) for k, v in retrieval_scores.items()}
