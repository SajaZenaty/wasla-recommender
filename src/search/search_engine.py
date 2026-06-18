
import numpy as np
from datetime import datetime
from src.data.preprocessing import clean_text_for_transformer
from src.features.embeddings import EmbeddingModel
from functools import lru_cache

@lru_cache(maxsize=128)
def get_query_embedding(query):
    model = EmbeddingModel.get_model()
    clean_query = clean_text_for_transformer(query)
    return model.encode([clean_query], normalize_embeddings=True)

def search_posts(query, system_data, top_k=20, threshold=0.4):
    query_vec = get_query_embedding(query)
    
    distances, indices = system_data["index"].search(query_vec.astype('float32'), top_k)
    
    results = []
    now = datetime.now()
    
    for i in range(len(indices[0])):
        idx = indices[0][i]
        sim_score = float(distances[0][i])
        
        if sim_score < threshold:
            continue
            
        post_id = system_data["idx_to_post_id"][idx]
        post = system_data["posts_by_id"].loc[post_id]
        post_time = post.get("timestamp", now)
        days_diff = (now - post_time).days
        freshness = float(np.exp(-max(0, days_diff) / 7)) 
        
        author_id = post.get("user_id")
        author_trust = system_data["user_trust_map"].get(author_id, 0.0)
        trust_score = author_trust / 5.0 
      
        final_score = (0.5 * sim_score) + (0.3 * freshness) + (0.2 * trust_score)
        
        results.append({
            "post_id": post_id,
            "similarity_score": sim_score,
            "freshness": round(freshness, 3),
            "trust": round(trust_score, 3),
            "final_score": round(final_score, 4)
        })
    
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
    return results