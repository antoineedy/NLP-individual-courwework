import pandas as pd
import numpy as np

# random forest

from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    return clf
