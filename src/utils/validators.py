import pandas as pd

from src.data.preprocessing import INTERACTION_COLUMNS

VALID_ACTIONS = {"click", "save", "apply"}


def validate_posts_data(df):

    required_columns = ["post_id", "user_id", "category", "title", "description"]
  
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Data Error: Column '{col}' is missing from the posts DataFrame.")
            
 
    critical_columns = ["post_id", "user_id", "category"]
    if df[critical_columns].isnull().any().any():
        raise ValueError("Data Error: Found null values in critical columns (post_id, user_id, category).")
        
    print(" Posts data validation passed.")
    return True



def validate_users_data(df):

    required_columns = ["user_id", "skills", "needs"]
  
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Data Error: Column '{col}' is missing from the users DataFrame.")
            
 
    critical_columns = ["user_id", "skills", "needs"]
    if df[critical_columns].isnull().any().any():
        raise ValueError("Data Error: Found null values in critical columns (user_id, skills , needs).")
        
    print(" Users data validation passed.")
    return True


def validate_interactions_data(df):
    if df.empty:
        return True

    for col in INTERACTION_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Data Error: Column '{col}' is missing from the interactions DataFrame."
            )

    critical_columns = ["user_id", "post_id", "action"]
    if df[critical_columns].isnull().any().any():
        raise ValueError(
            "Data Error: Found null values in critical interaction columns "
            "(user_id, post_id, action)."
        )

    invalid_actions = set(df["action"].astype(str)) - VALID_ACTIONS
    if invalid_actions:
        raise ValueError(
            "Data Error: Invalid interaction actions: "
            f"{sorted(invalid_actions)}. Expected click, save, or apply."
        )

    return True