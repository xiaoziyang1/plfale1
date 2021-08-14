import Base


class BasePLECOC(Base.Base):
    def __init__(self, params):
        self.estimator = params.get("classifier")
        self.params = params
        self.is_ecoc = True
        self.bin_pre = None
        self.co_exist = None

    def fit(self, train_data, tarin_labels):
        pass

    def predict(self, test_data):
        pass
