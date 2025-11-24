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
    X_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]
    outputs: npt.NDArray[np.float64]
    keys: list[str]
    auto_predict: bool

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self):
        if self.X_train is None or self.y_train is None:
            self.load_train_data()
        if self.X_test is None or self.y_test is None:
            self.load_test_data()

        self.train(self.X_train, self.y_train)
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

    def mae(self, key: str, y_test=None, output=None) -> float | list[float]:
        if y_test is None:
            y_test = self.y_test

        if output is None:
            output = self.outputs

        i = self.key_index(key)
        mae = mean_absolute_error(y_test[:, i], output[:, i])
        return mae

    def rmse(self, key: str, y_test=None, output=None) -> float | list[float]:
        if y_test is None:
            y_test = self.y_test

        if output is None:
            output = self.outputs

        i = self.key_index(key)
        rmse = mean_squared_error(y_test[:, i], output[:, i], squared=False)
        return rmse

    def set_train_data(self, X, y, keys, shape):
        self.X_train, self.y_train, self.keys, self.shape = X, y, keys, shape

    def set_test_data(self, X, y, keys, shape):
        self.X_test, self.y_test, self.keys, self.shape = X, y, keys, shape

    def load_train_data(self):
        self.X_train, self.y_train, self.keys, self.shape = get_time_data(
            config=self.config, file_keyword=self.config.train_year
        )

    def load_test_data(self):
        self.X_test, self.y_test, self.keys, self.shape = get_time_data(
            config=self.config, file_keyword=self.config.test_year
        )

    def get_train_data(self) -> Any:
        return self.X_train, self.y_train, self.keys, self.shape

    def get_test_data(self) -> Any:
        return self.X_test, self.y_test, self.keys, self.shape
