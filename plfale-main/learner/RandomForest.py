import learner.BaseLearner as BaseLearner
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForest(BaseLearner):

    def __init__(self, params):
        BaseLearner.__init__(self, params)
        self.model = RandomForestClassifier(n_estimators=10)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        p_labels = self.model.predict(X)
        return p_labels

    def predict_proba(self, X):
        p_vals = self.model.predict_proba(X)
        p_labels = np.argmax(p_vals, axis=1)
        p_labels[p_labels == 0] = -1
        p_vals = np.max(p_vals, axis=1)
        return p_labels, p_vals
