import re
import pandas as pd

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
    
    df["desc_clean"] = df["description"].apply(clean_text_for_transformer).fillna("")
    df["title_clean"] = df["title"].apply(clean_text_for_transformer).fillna("")
    
    df["category"] = df["category"].astype(str).str.strip().str.lower().fillna("unknown")
    df["location"] = df["location"].astype(str).str.strip().str.lower().fillna("unknown")
    
    df["full_text"] = (
        df["title_clean"] + " " + 
        df["desc_clean"] + " " + 
        df["category"]
    )
    
    return df

def preprocess_users(users_df):
    df = users_df.copy()
    
    df["skills"] = df["skills"].apply(lambda x: x if isinstance(x, list) else [])
    df["needs"] = df["needs"].apply(lambda x: x if isinstance(x, list) else [])
    
    df["skills"] = df["skills"].apply(lambda lst: [str(x).lower().strip() for x in lst])
    df["needs"] = df["needs"].apply(lambda lst: [str(x).lower().strip() for x in lst])
    
    df["location"] = df["location"].fillna("unknown").astype(str).str.lower().str.strip()
    
    df["time_balance"] = pd.to_numeric(df["time_balance"], errors='coerce').fillna(0)
    
    return df