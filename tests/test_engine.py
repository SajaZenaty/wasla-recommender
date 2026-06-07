from src.data.loader import load_mock_data
from src.recommender.engine import bootstrap_system_data, recommend


def test_recommend_returns_ranked_eligible_posts():
    users_df, posts_df, interactions_df = load_mock_data(n_users=8, seed=42)
    system_data = bootstrap_system_data(users_df, posts_df, interactions_df)

    user = users_df.iloc[0].to_dict()
    results = recommend(user, posts_df, interactions_df, system_data, top_k=5)

    assert len(results) <= 5
    if not results.empty:
        scores = results["final_score"].tolist()
        assert scores == sorted(scores, reverse=True)
        # A user should never be recommended their own posts.
        assert (results["user_id"] != user["user_id"]).all()
        # Eligible posts must be affordable.
        assert (results["time_credits"] <= user["time_balance"]).all()
