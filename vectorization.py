import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class Vectorization:
    def __init__(self, method, X_train=None):
        self.method = method
        self.X_train = X_train
        self.vectorizer = None
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(self.X_train)
        elif self.method == "count":
            self.vectorizer = CountVectorizer()
            self.vectorizer.fit(self.X_train)
        elif self.method == None:
            print("Nothing to do here")
        else:
            raise ValueError("Invalid method")

    def vectorize(self, X_test):
        if self.method == "tfidf":
            return self.vectorizer.transform(X_test)
        elif self.method == "count":
            return self.vectorizer.transform(X_test)
        elif self.method == None:
            return X_test
