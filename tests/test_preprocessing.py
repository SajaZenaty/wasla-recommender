import pandas as pd
import pytest

from src.data.express_loader import frames_from_payload
from src.data.preprocessing import INTERACTION_COLUMNS, ensure_interactions_schema, preprocess_posts, preprocess_users
from src.utils.validators import validate_interactions_data


def test_ensure_interactions_schema_empty_list():
    df = ensure_interactions_schema(pd.DataFrame([]))
    assert list(df.columns) == INTERACTION_COLUMNS
    assert df.empty


def test_frames_from_payload_accepts_empty_interactions():
    users_df, posts_df, interactions_df = frames_from_payload(
        {
            "users": [{"user_id": 1, "skills": ["برمجة"], "needs": ["تصميم"]}],
            "posts": [
                {
                    "post_id": 10,
                    "user_id": 2,
                    "category": "برمجة",
                    "title": "خدمة",
                    "description": "وصف",
                }
            ],
            "interactions": [],
        }
    )

    assert list(interactions_df.columns) == INTERACTION_COLUMNS
    assert interactions_df.empty
    assert len(users_df) == 1
    assert len(posts_df) == 1


def test_preprocess_fills_express_optional_fields():
    users = preprocess_users(pd.DataFrame({"user_id": [1], "skills": [["a"]], "needs": [["b"]]}))
    posts = preprocess_posts(
        pd.DataFrame(
            {
                "post_id": [1],
                "user_id": [2],
                "category": ["x"],
                "title": ["t"],
                "description": ["d"],
            }
        )
    )

    assert users.loc[0, "trust_score"] == 0
    assert posts.loc[0, "time_credits"] == 0
    assert posts.loc[0, "post_type"] == "عرض"


def test_validate_interactions_rejects_invalid_action():
    df = pd.DataFrame(
        {
            "user_id": [1],
            "post_id": [2],
            "action": ["like"],
            "timestamp": [pd.Timestamp("2024-01-01")],
        }
    )
    with pytest.raises(ValueError, match="Invalid interaction actions"):
        validate_interactions_data(df)
