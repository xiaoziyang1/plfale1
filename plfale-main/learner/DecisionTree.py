import learner.BaseLearner as BaseLearner
from sklearn import tree
import numpy as np


class DecisionTree(BaseLearner):

    def __init__(self, params):
        BaseLearner.__init__(self, params)
        self.model = tree.DecisionTreeClassifier()

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
