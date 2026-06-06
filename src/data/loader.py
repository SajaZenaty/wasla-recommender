import pandas as pd

from scripts.generate_users import generate_users
from scripts.generate_posts import generate_posts
from scripts.generate_interactions import generate_interactions
from src.data.preprocessing import preprocess_posts, preprocess_users


def load_mock_data(n_users=50, seed=None):
    users_df = generate_users(n_users)
    posts_df = generate_posts(users_df)
    interactions_df = generate_interactions(users_df, posts_df, seed=seed)

    users_df = preprocess_users(users_df)
    posts_df = preprocess_posts(posts_df)

    return users_df, posts_df, interactions_df
