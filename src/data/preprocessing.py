import re
import pandas as pd

INTERACTION_COLUMNS = ["user_id", "post_id", "action", "timestamp"]
USER_COLUMNS = ["user_id", "skills", "needs", "time_balance", "trust_score"]


def ensure_interactions_schema(interactions_df):
    """Return an interactions frame with the expected columns.

    ``pd.DataFrame([])`` has no columns; bootstrap payloads often send an empty
    interactions list before any user activity exists.
    """
    if interactions_df is None:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    if interactions_df.empty and not set(INTERACTION_COLUMNS).issubset(interactions_df.columns):
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    return interactions_df


def ensure_users_schema(users_df):
    """Return a users frame with the expected columns (may be empty)."""
    if users_df is None:
        return pd.DataFrame(columns=USER_COLUMNS)
    if users_df.empty and not {"user_id", "skills", "needs"}.issubset(users_df.columns):
        return pd.DataFrame(columns=USER_COLUMNS)
    return users_df


def _ensure_column(df, column, default):
    if column not in df.columns:
        df[column] = default
    return df

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    return text


def clean_text_for_transformer(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = normalize_arabic(text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = " ".join(text.split())

    return text




def preprocess_posts(posts_df):
    df = posts_df.copy()

    
    _ensure_column(df, "time_credits", 0)
    _ensure_column(df, "post_type", "offer")
    _ensure_column(df, "service_mode", None)

    df["desc_clean"] = df["description"].apply(clean_text_for_transformer).fillna("")
    df["title_clean"] = df["title"].apply(clean_text_for_transformer).fillna("")

    df["category"] = df["category"].astype(str).str.strip().str.lower().fillna("unknown")
    df["time_credits"] = pd.to_numeric(df["time_credits"], errors="coerce").fillna(0)

    df["full_text"] = (
        df["title_clean"] + " " +
        df["desc_clean"] + " " +
        df["category"]
    )

    return df


def preprocess_users(users_df):
    df = users_df.copy()

    _ensure_column(df, "time_balance", 0)
    _ensure_column(df, "trust_score", 0)

    df["skills"] = df["skills"].apply(lambda x: x if isinstance(x, list) else [])
    df["needs"] = df["needs"].apply(lambda x: x if isinstance(x, list) else [])

    df["skills"] = df["skills"].apply(lambda lst: [str(x).lower().strip() for x in lst])
    df["needs"] = df["needs"].apply(lambda lst: [str(x).lower().strip() for x in lst])

    df["time_balance"] = pd.to_numeric(df["time_balance"], errors="coerce").fillna(0)
    df["trust_score"] = pd.to_numeric(df["trust_score"], errors="coerce").fillna(0)

    return df
