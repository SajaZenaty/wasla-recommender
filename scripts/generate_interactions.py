import random
import pandas as pd
from datetime import datetime, timedelta

ACTIONS = ["click", "save", "apply"]
ACTION_PROBS = [0.6, 0.3, 0.1]


def generate_interactions(users_df, posts_df, seed=None):
    if seed is not None:
        random.seed(seed)

    interactions = []
    now = datetime.now()
    post_ids = posts_df["post_id"].tolist()
    posts_by_user = posts_df.groupby("user_id")["post_id"].apply(set).to_dict()

    for _, user in users_df.iterrows():
        user_id = user["user_id"]

        if random.random() < 0.3:
            continue

        own_posts = posts_by_user.get(user_id, set())
        eligible = [p for p in post_ids if p not in own_posts]
        if not eligible:
            continue

        n_interactions = random.randint(5, 20)
        sampled_posts = random.sample(eligible, min(n_interactions, len(eligible)))

        for post_id in sampled_posts:
            action = random.choices(ACTIONS, weights=ACTION_PROBS, k=1)[0]
            days_ago = random.randint(0, 30)
            interactions.append({
                "user_id": user_id,
                "post_id": post_id,
                "action": action,
                "timestamp": now - timedelta(days=days_ago),
            })

    return pd.DataFrame(interactions)
