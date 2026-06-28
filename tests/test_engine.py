from src.data.loader import load_mock_data
from src.data.preprocessing import is_offer_post
from src.features.embeddings import get_user_vector
from src.ranking.scoring import compute_similarity
from src.recommender.engine import bootstrap_system_data, recommend
from tests.fixtures.sample_data import load_sample_frames


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


def test_compute_similarity_matches_needs_for_arabic_offers():
    user = {"needs": ["برمجة"], "skills": ["تصميم"]}
    need_match = {"post_type": "عرض", "category": "برمجة"}
    skill_only = {"post_type": "عرض", "category": "تصميم"}

    assert compute_similarity(user, need_match) == 1.0
    assert compute_similarity(user, skill_only) == 0.2


def test_get_user_vector_uses_consumer_for_arabic_offers():
    consumer = object()
    provider = object()
    user_vectors = {"consumer": consumer, "provider": provider}

    assert get_user_vector(user_vectors, {"post_type": "عرض"}) is consumer
    assert get_user_vector(user_vectors, {"post_type": "طلب"}) is provider


def test_recommend_top_offer_matches_user_needs():
    users_df, posts_df, interactions_df = load_sample_frames()
    system_data = bootstrap_system_data(users_df, posts_df, interactions_df)

    user = users_df.iloc[0].to_dict()
    results = recommend(user, posts_df, interactions_df, system_data, top_k=5)

    assert not results.empty
    offer_results = results[results["post_type"].apply(is_offer_post)]
    assert not offer_results.empty
    user_needs = {n.lower().strip() for n in user["needs"]}
    top_offer = offer_results.iloc[0]
    assert str(top_offer["category"]).lower().strip() in user_needs
