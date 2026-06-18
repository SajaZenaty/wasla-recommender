import numpy as np
from src.recommender.engine import recommend

def calculate_metrics(recommended, relevant, k=10):
    """حساب Precision و Recall لمستخدم واحد."""
    if not relevant:
        return 0.0, 0.0
    
    top_k = recommended[:k]
    intersection = set(top_k) & set(relevant)
    
    precision = len(intersection) / k
    recall = len(intersection) / len(relevant)
    
    return precision, recall

def evaluate_recommender(users_df, posts_df, interactions_df, system_data, top_k=10):
    """
    التقييم الحقيقي: نمرر system_data لضمان عمل الـ recommend.
    """
    precisions = []
    recalls = []
    
    # استخراج التفاعلات الفعلية
    ground_truth = interactions_df.groupby("user_id")["post_id"].apply(list).to_dict()
    
    for _, user_row in users_df.iterrows():
        user_id = user_row["user_id"]
        if user_id not in ground_truth:
            continue
            
        # نمرر system_data التي تم بناؤها مسبقاً
        results = recommend(
            user=user_row.to_dict(), 
            posts_df=posts_df, 
            interactions_df=interactions_df, 
            system_data=system_data, 
            top_k=top_k
        )
        
        if results.empty:
            continue
            
        recommended_ids = results["post_id"].tolist()
        relevant_ids = ground_truth[user_id]
        
        p, r = calculate_metrics(recommended_ids, relevant_ids, k=top_k)
        precisions.append(p)
        recalls.append(r)
        
    return {
        "precision": np.mean(precisions) if precisions else 0.0,
        "recall": np.mean(recalls) if recalls else 0.0
    }