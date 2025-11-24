from typing import override

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

    def get_coefficients(self):
        return self.lr.coef_
        pass

    @override
    def train(self, X=None, y=None):
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train
        p = PolynomialFeatures(self.degree)
        X_p = p.fit_transform(X)
        self.train_time, _ = track_time(lambda: self.lr.fit(X_p, y))

    @override
    def predict(self, X=None):
        if X is None:
            X = self.X_test
        p = PolynomialFeatures(self.degree)
        X_p = p.transform(X)
        self.predict_time, self.outputs = track_time(lambda: self.lr.predict(X_p))
