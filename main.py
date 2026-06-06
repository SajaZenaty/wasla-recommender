from src.data.loader import load_mock_data
from src.recommender.engine import bootstrap_system_data, recommend
from src.evaluation.offline_eval import evaluate_recommender, print_eval_summary


def main():
    users_df, posts_df, interactions_df = load_mock_data(seed=42)
    system_data = bootstrap_system_data(users_df, posts_df, interactions_df)

    sample_users = users_df.head(3)

    for _, user_row in sample_users.iterrows():
        user = user_row.to_dict()
        results = recommend(user, posts_df, interactions_df, system_data, top_k=10)

        print(f"\n{'=' * 60}")
        print(f"User {user['user_id']} | needs: {user['needs']} | skills: {user['skills']}")
        print(f"{'=' * 60}")

        if results.empty:
            print("No recommendations found.")
            continue

        display_cols = [
            "title", "category", "post_type", "final_score",
            "semantic", "retrieval", "cf", "category_score",
            "location", "time_fit", "freshness", "trust", "balance_bias",
        ]
        print(results[display_cols].to_string(index=False))

    summary = evaluate_recommender(n_users=50, top_k=10, seeds=(42, 43, 44))
    print_eval_summary(summary)


if __name__ == "__main__":
    main()
