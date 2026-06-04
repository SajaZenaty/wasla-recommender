import re

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