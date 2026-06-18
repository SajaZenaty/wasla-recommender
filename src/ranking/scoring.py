import numpy as np
from datetime import datetime
from src.data.preprocessing import normalize_arabic
from src import settings

def _normalize_category(value):
    return normalize_arabic(str(value).lower().strip()) if value else ""

def compute_similarity(user, post):
    user_skills = {_normalize_category(s) for s in user.get("skills", [])}
    user_needs = {_normalize_category(n) for n in user.get("needs", [])}
    post_category = _normalize_category(post.get("category", ""))
    post_type = post.get("post_type") 

    if post_type == "عرض":
        return 1.0 if post_category in user_needs else 0.0
    elif post_type == "طلب":
        return 1.0 if post_category in user_skills else 0.0
    
    return 0.0

def compute_time_fit(user, post):
    balance = user.get("time_balance", 0)
    cost = post.get("time_credits", 0)
    if cost > balance: return 0.0
    surplus = balance - cost
    return min(1.0, 0.5 + (surplus / (surplus + 5)))

def location_score(user, post):
    if post.get("service_mode") == "الكتروني": return 1.0
    return 1.0 if _normalize_category(user.get("location", "")) == _normalize_category(post.get("location", "")) else 0.3

def freshness_score(post):
    post_time = post.get("timestamp")
    if not post_time: return 0.5
    days = (datetime.now() - post_time).days
    return float(np.exp(-max(0, days) / 7))

def compute_time_balance_bias(user, post):
    balance = user.get("time_balance", 0)
    post_type = post.get("post_type")
    if balance < 5:
        return 0.3 if post_type == "طلب" else -0.05
    
    elif balance > 20:
        return 0.3 if post_type == "عرض" else -0.05

    return 0.0

def trust_bonus(author_trust, max_trust=5.0):
    return 0.02 * (author_trust / max_trust)


from typing import Tuple, Dict

def compute_hybrid_score(user: Dict, post: Dict, user_vec, post_vec, cf_score: float, retrieval_prior: float, author_trust: float) -> Tuple[float, Dict]:
    
    components = {
        "semantic": float(np.dot(user_vec, post_vec)),
        "retrieval": retrieval_prior,
        "cf": cf_score,
        "category": compute_similarity(user, post),
        "location": location_score(user, post),
        "time_fit": compute_time_fit(user, post),
        "freshness": freshness_score(post),
        "trust": trust_bonus(author_trust)
    }
    
    balance_bias = compute_time_balance_bias(user, post)
    
    final_score = sum(components[k] * settings.SCORING_WEIGHTS.get(k, 0) for k in components) + balance_bias
    
    breakdown = {k: round(v, 4) for k, v in components.items()}
    breakdown["balance_bias"] = round(balance_bias, 4)
    breakdown["final_score"] = round(final_score, 4)
    
    return final_score, breakdown