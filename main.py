import time
import logging

from src.data.loader import load_mock_data
from src.recommender.engine import bootstrap_system_data, recommend
from src.evaluation.offline_eval import evaluate_precision_recall, evaluate_recommender, print_eval_summary
from src.search.search_engine import search_posts

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    try:
        logging.info("Starting System Bootstrap...")
        start_time = time.time()

        users_df, posts_df, interactions_df = load_mock_data(seed=42)
        system_data = bootstrap_system_data(users_df, posts_df, interactions_df)

        logging.info(f"Bootstrap finished in {time.time() - start_time:.2f} seconds.")

        sample_users = users_df.head(3)

        for _, user_row in sample_users.iterrows():
            user = user_row.to_dict()

            rec_start = time.time()
            results = recommend(user, posts_df, interactions_df, system_data, top_k=10)
            rec_end = time.time()

            print(f"\n{'=' * 60}")
            print(f"User {user['user_id']} | Needs: {user['needs']} | Skills: {user['skills']}")
            print(f"Recommendation generated in {rec_end - rec_start:.4f}s")
            print(f"{'=' * 60}")

            if results.empty:
                print("No recommendations found.")
                continue

            display_cols = [
                "title", "category", "final_score", "semantic", "cf",
                "location_score", "trust",
            ]
            existing_cols = [c for c in display_cols if c in results.columns]
            print(results[existing_cols].to_string(index=False))

        logging.info("\nRunning search demo...")
        search_results = search_posts("برمجة", system_data, top_k=5)
        for hit in search_results:
            print(hit)

        logging.info("\nRunning offline evaluation...")
        summary = evaluate_recommender()
        print_eval_summary(summary)

        pr_summary = evaluate_precision_recall(
            users_df, posts_df, interactions_df, system_data, top_k=10
        )
        print(f"\nPrecision/Recall: {pr_summary}")

    except Exception as e:
        logging.error(f"Critical System Failure: {e}")


if __name__ == "__main__":
    main()
