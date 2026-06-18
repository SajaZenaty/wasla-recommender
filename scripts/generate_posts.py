import random
import pandas as pd
from datetime import datetime, timedelta


category_variants = {
    "برمجة": ["برمجة", "programming", "coding", "web development"],
    "تصميم": ["تصميم", "design", "UX UI"],
    "تعليم": ["تعليم", "teaching"],
    "تصوير": ["تصوير", "photography"],
    "تنظيف": ["تنظيف", "cleaning"],
    "طبخ": ["طبخ", "cooking"],
    "ترجمة": ["ترجمة", "translation"],
    "صيانة": ["صيانة", "maintenance"],
}


description_templates_offer = [
    "أقدم خدمات {skill}",
    "خبرة في {skill}",
    "محترف {skill}",
    "Professional in {skill}",
]


description_templates_request = [
    "أحتاج مساعدة في {skill}",
    "Need help with {skill}",
    "بدور على حد يفهم في {skill}",
]


def generate_posts(users_df):
    posts = []
    post_id = 0
    now = datetime.now()

    for _, user in users_df.iterrows():

        for skill in user["skills"]:
            skill_variant = random.choice(category_variants[skill])

            post_time = now - timedelta(days=random.randint(0, 14))

            posts.append({
                "post_id": post_id,
                "user_id": user["user_id"],
                "post_type": "عرض",
                "category": skill,
                "title": f"خدمة {skill_variant}",
                "description": random.choice(description_templates_offer).format(skill=skill_variant),
                "service_mode": random.choice(["الكتروني", "وجاهي"]),
                "location": user["location"],
                "time_credits": random.randint(1, 5),
                "timestamp": post_time
            })

            post_id += 1

        for need in user["needs"]:
            need_variant = random.choice(category_variants[need])

            post_time = now - timedelta(days=random.randint(0, 14))

            posts.append({
                "post_id": post_id,
                "user_id": user["user_id"],
                "post_type": "طلب",
                "category": need,
                "title": f"طلب {need_variant}",
                "description": random.choice(description_templates_request).format(skill=need_variant),
                "service_mode": random.choice(["الكتروني", "وجاهي"]),
                "location": user["location"],
                "time_credits": random.randint(1, 5),
                "timestamp": post_time
            })

            post_id += 1

    posts_df = pd.DataFrame(posts)

    return posts_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)