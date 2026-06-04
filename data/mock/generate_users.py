import random
import pandas as pd

categories = [
    "تعليم", "برمجة", "تصميم", "صيانة",
    "ترجمة", "طبخ", "تنظيف", "تصوير"
]

category_weights = [15, 25, 20, 10, 10, 5, 5, 5]

locations = [
    "غزة", "خانيونس", "النصيرات", "المغازي",
    "البريج", "رفح", "جباليا", "دير البلح"
]


def generate_users(n_users=50):
    users = []

    for i in range(n_users):
        num_skills = random.randint(1, 5)
        num_needs = random.randint(1, 5)

        skills = list(set(random.choices(categories, weights=category_weights, k=num_skills)))

        if random.random() < 0.3:
            needs = random.sample(categories, min(num_needs, len(categories)))
        else:
            remaining = [c for c in categories if c not in skills]
            needs = random.sample(remaining, min(num_needs, len(remaining)))

        users.append({
            "user_id": i,
            "skills": skills,
            "needs": needs,
            "location": random.choice(locations),
            "time_balance": random.randint(0, 20),
            "trust_score": round(random.uniform(0, 5)),
        })

    return pd.DataFrame(users)

