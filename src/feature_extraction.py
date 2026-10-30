import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------
# TF-IDF Model
# -------------------------
tfidf = TfidfVectorizer(max_features=3000)


def extract_url_features(urls):
    count = len(urls)

    suspicious_keywords = ["login", "verify", "secure", "bank", "update"]

    suspicious = 0
    for url in urls:
        if any(word in url.lower() for word in suspicious_keywords):
            suspicious += 1

    return [count, suspicious]


def extract_text_features(text):

    urgent_words = ["urgent", "immediately", "verify", "password", "click", "limited"]

    words = text.lower().split()

    urgent_count = sum(1 for w in words if w in urgent_words)

    return [len(words), urgent_count]


def fit_tfidf(corpus):
    return tfidf.fit(corpus)


def transform_tfidf(corpus):
    return tfidf.transform(corpus).toarray()