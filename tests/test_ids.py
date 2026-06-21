import pandas as pd
import pytest

from src.api.state import RecommenderState
from src.data.preprocessing import preprocess_posts, preprocess_users
from src.utils.ids import filter_by_id, lookup_in_mapping


def test_filter_by_id_accepts_string_for_int_column():
    df = pd.DataFrame({"user_id": [1, 2], "name": ["a", "b"]})
    matches = filter_by_id(df, "user_id", "1")
    assert len(matches) == 1
    assert matches.iloc[0]["name"] == "a"


def test_lookup_in_mapping_accepts_string_for_int_key():
    mapping = {1: "found"}
    assert lookup_in_mapping(mapping, "1") == "found"
    assert lookup_in_mapping(mapping, 99) is None


def test_set_data_rolls_back_on_rebuild_failure():
    state = RecommenderState()
    good_users = preprocess_users(
        pd.DataFrame({"user_id": [1], "skills": [["a"]], "needs": [["b"]], "trust_score": [1]})
    )
    good_posts = preprocess_posts(
        pd.DataFrame(
            {
                "post_id": [10],
                "user_id": [2],
                "post_type": ["عرض"],
                "category": ["x"],
                "title": ["t"],
                "description": ["d"],
                "time_credits": [1],
                "location": ["g"],
            }
        )
    )
    empty_interactions = pd.DataFrame(columns=["user_id", "post_id", "action", "timestamp"])
    state.set_data(good_users, good_posts, empty_interactions)

    bad_posts = pd.DataFrame({"post_id": [10]})
    with pytest.raises(KeyError):
        state.set_data(good_users, bad_posts, empty_interactions)

    assert state.ready is True
    assert state.post_count == 1
