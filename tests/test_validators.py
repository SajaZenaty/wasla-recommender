import pandas as pd
import pytest

from src.utils.validators import validate_posts_data, validate_users_data


def test_posts_validator_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_posts_data(pd.DataFrame({"post_id": [1]}))


def test_users_validator_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_users_data(pd.DataFrame({"user_id": [1]}))


def test_validators_accept_well_formed_frames():
    posts = pd.DataFrame(
        {
            "post_id": [1],
            "user_id": [2],
            "category": ["برمجة"],
            "title": ["خدمة"],
            "description": ["وصف"],
        }
    )
    users = pd.DataFrame({"user_id": [2], "skills": [["برمجة"]], "needs": [["تصميم"]]})

    assert validate_posts_data(posts) is True
    assert validate_users_data(users) is True
