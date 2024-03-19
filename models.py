import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForTokenClassification


class Models:
    """
    random_forest, pretrained-bert
    """

    def __init__(self, method, X_train=None, y_train=None):
        self.method = method
        self.model = None
        if self.method == "random_forest":
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
        elif self.method == "pretrained-bert":
            self.model = AutoModelForTokenClassification.from_pretrained(
                "surrey-nlp/roberta-base-finetuned-abbr"
            )
        else:
            raise ValueError("Invalid method")

    def predict(self, X_test):
        if self.method == "random_forest":
            return self.model.predict(X_test)
        elif self.method == "pretrained-bert":
            inputs = X_test
            labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(
                0
            )  # Batch size 1
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            out = torch.argmax(logits, dim=2)
            out
            return out
