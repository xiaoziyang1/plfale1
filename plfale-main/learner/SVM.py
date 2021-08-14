import learner.BaseLearner as BaseLearner
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from svmutil import *


class SVM(BaseLearner):

    def __init__(self, params):
        BaseLearner.__init__(self, params)
        self.model = None
        self.isBigData = None
        self.isMultiThread = self.params.get('pool') is not None

    def fit(self, X, Y):
        # prob = svm_problem(Y, X)
        # param = svm_parameter(self.params.get('svm_param'))
        # self.model = svm_train(prob, param)
        # self.isBigData = len(X) > 10000
        if self.isMultiThread:  # libsvm在多线程下会报错，原因不明
            self.isBigData = False
            if self.isBigData:
                self.model = LinearSVC()
            else:
                self.model = SVC(gamma='auto', probability=True)
            self.model.fit(X, Y)
        else:
            prob = svm_problem(Y, X)
            param = svm_parameter(self.params.get('svm_param'))
            self.model = svm_train(prob, param)

    def predict(self, X):
        # test_label_vector = np.ones(len(X))
        # p_labels, _, _ = svm_predict(test_label_vector, X, self.model, options='-q')
        if self.isMultiThread:
            p_labels = self.model.predict(X)
        else:
            p_labels, _, _ = svm_predict(np.ones(len(X)).tolist(), X, self.model, options='-q')
        return p_labels

    def predict_proba(self, X):
        # test_label_vector = np.ones(len(X))
        # p_labels, _, p_vals = svm_predict(test_label_vector, X, self.model, options='-q')
        # return p_labels, p_vals
        if self.isMultiThread:
            if self.isBigData:
                p_vals = self.model.decision_function(X)
                p_labels = np.sign(p_vals)
            else:
                p_vals = self.model.decision_function(X)
                p_labels = np.sign(p_vals)
                # p_vals = np.max(p_vals, axis=1)
        else:
            p_labels, _, p_vals = svm_predict(np.ones(len(X)).tolist(), X, self.model, options='-q')
        return p_labels, p_vals
