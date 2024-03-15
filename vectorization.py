import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(X_train):
    X_train = pd.Series(X_train)
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(X_train)
    return vectorizer


def count(X_train):
    X_train = pd.Series(X_train)
    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(X_train)
    return vectorizer
