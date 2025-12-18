from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from cuml.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

from aggregate import get_time_data
from config import Config


class MeteoData:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Model(ABC):
    train_time: float
    predict_time: float
    keys: list[str]
    auto_predict: bool

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self):
        if self.X_train is None or self.y_train is None:
            self.load_train_data()
        if self.X_test is None or self.y_test is None:
            self.load_test_data()

        print("TRAIN")
        self.train(self.X_train, self.y_train)
        print("PREDICT")
        self.predict(self.X_test)

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def get_train_time(self) -> float:
        return self.train_time

    def get_predict_time(self) -> float:
        return self.predict_time

    def key_index(self, key: str):
        return self.keys.index(key)

    def mae(self, key: str, y_test, output) -> float:
        i = self.key_index(key)
        mae = mean_absolute_error(y_test[:, i], output[:, i])
        return mae

    def rmse(self, key: str, y_test, output) -> float:
        i = self.key_index(key)
        rmse = mean_squared_error(y_test[:, i], output[:, i], squared=False)
        return rmse

    def squared_error(self, key, y_test, output) -> float:
        i = self.key_index(key)
        return sum((y_test[:, i] - output[:, i]) ** 2)

    def rmse_skill(self, key: str, ref, y_test, output) -> float:
        rmse_pred = self.rmse(key, output, y_test)
        rmse_ref = self.rmse(key, ref, y_test)
        skill = 1.0 - (rmse_pred / rmse_ref)
        return skill
