import re

URL_PATTERN = r'https?://\S+|www\.\S+'


def clean_text(text):
    text = str(text).lower()
    text = re.sub(URL_PATTERN, '', text)
    text = re.sub(r'\s*\n+\s*', ' ', text)
    return text.strip()