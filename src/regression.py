from cuml.preprocessing import PolynomialFeatures

from model import Config, Model
from utils import track_time


class Regression(Model):
    def __init__(self, lr, degree, config: Config):
        super().__init__(config)
        self.degree = degree
        self.lr = lr

    def name(self) -> str:
        return "Linear Regression"

    def run(self):
        self.X_train, self.y_train, train_keys, train_shape = super().train_data()
        self.X_test, self.y_test, test_keys, test_shape = super().test_data()

        self.keys = train_keys
        self.shape = train_shape

        p = PolynomialFeatures(self.degree)
        X_train_p = p.fit_transform(self.X_train)
        X_test_p = p.transform(self.X_test)

        self.train_time, _ = track_time(lambda: self.lr.fit(X_train_p, self.y_train))

        self.predict_time, self.output = track_time(lambda: self.lr.predict(X_test_p))

    def get_coefficients(self):
        return self.lr.coef_
        pass
