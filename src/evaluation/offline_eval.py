"""Lightweight offline evaluation over the mock dataset.

These metrics are intended for local sanity checks and CI, not for the
production serving path. They run the full recommender for every generated
user across a few seeds and report simple aggregate quality signals.
"""
import numpy as np

from src.data.loader import load_mock_data
from src.recommender.engine import bootstrap_system_data, recommend


def _category_hit(user, row):
    needs = {str(n).lower().strip() for n in user.get("needs", [])}
    skills = {str(s).lower().strip() for s in user.get("skills", [])}
    category = str(row.get("category", "")).lower().strip()
    if row.get("post_type") == "عرض":
        return category in needs
    return category in skills


def _evaluate_seed(seed, n_users, top_k):
    users_df, posts_df, interactions_df = load_mock_data(n_users=n_users, seed=seed)
    system_data = bootstrap_system_data(users_df, posts_df, interactions_df)

    total_recs = 0
    total_hits = 0
    score_sum = 0.0
    recommended_posts = set()
    users_with_recs = 0

    for _, user_row in users_df.iterrows():
        user = user_row.to_dict()
        results = recommend(user, posts_df, interactions_df, system_data, top_k=top_k)
        if results.empty:
            continue
        users_with_recs += 1
        for _, row in results.iterrows():
            total_recs += 1
            score_sum += float(row.get("final_score", 0.0))
            recommended_posts.add(row.get("post_id"))
            if _category_hit(user, row):
                total_hits += 1

    catalog_size = len(posts_df)
    return {
        "seed": seed,
        "avg_final_score": score_sum / total_recs if total_recs else 0.0,
        "category_hit_rate": total_hits / total_recs if total_recs else 0.0,
        "coverage": len(recommended_posts) / catalog_size if catalog_size else 0.0,
        "users_with_recs": users_with_recs,
        "n_users": len(users_df),
    }


def evaluate_recommender(n_users=50, top_k=10, seeds=(42, 43, 44)):
    per_seed = [_evaluate_seed(seed, n_users, top_k) for seed in seeds]

    def _avg(key):
        return float(np.mean([s[key] for s in per_seed])) if per_seed else 0.0

    return {
        "per_seed": per_seed,
        "avg_final_score": _avg("avg_final_score"),
        "category_hit_rate": _avg("category_hit_rate"),
        "coverage": _avg("coverage"),
        "top_k": top_k,
        "seeds": list(seeds),
    }


def print_eval_summary(summary):
    print(f"\n{'=' * 60}")
    print("OFFLINE EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Seeds: {summary['seeds']} | top_k: {summary['top_k']}")
    print(f"Avg final score  : {summary['avg_final_score']:.4f}")
    print(f"Category hit rate: {summary['category_hit_rate']:.4f}")
    print(f"Coverage         : {summary['coverage']:.4f}")
    for s in summary["per_seed"]:
        print(
            f"  seed={s['seed']} hit_rate={s['category_hit_rate']:.4f} "
            f"coverage={s['coverage']:.4f} avg_score={s['avg_final_score']:.4f} "
            f"users_with_recs={s['users_with_recs']}/{s['n_users']}"
        )
