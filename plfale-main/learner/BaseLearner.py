
class BaseLearner:

    def __init__(self, params):
        self.params = params

    def fit(self, train_data, tarin_labels):
        pass

    def predict(self, test_data):
        pass

    def predict_proba(self, test_data):
        pass
