from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from cuml.metrics import mean_absolute_error

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
    output: npt.NDArray[np.float64]
    keys: list[str]

    def __init__(self, config: Config) -> None:
        self.config = config

        if self.config.X_test is not None:
            self.X_test = self.config.X_test

        if self.config.X_train is not None:
            self.X_train = self.config.X_train

        if self.config.y_train is not None:
            self.y_train = self.config.y_train

        if self.config.y_test is not None:
            self.y_test = self.config.y_test

        if self.config.keys is not None:
            self.keys = self.config.keys

    @abstractmethod
    def run(self):
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
            output = self.output

        i = self.key_index(key)
        return mean_absolute_error(y_test[:, i], output[:, i])

    def train_data(self) -> Any:
        if self.X_train is not None and self.y_train is not None:
            return self.X_train, self.y_train, None, None
        else:
            self.X_train, self.y_train, self.keys, self.shape = get_time_data(
                config=self.config, file_keyword=self.config.train_year
            )
            return self.X_train, self.y_train, self.keys, self.shape

    def test_data(self) -> Any:
        if (
            self.X_train is not None
            and self.y_train is not None
            and self.keys is not None
        ):
            return self.X_test, self.y_test, self.keys, None
        else:
            self.X_test, self.y_test, self.keys, self.shape = get_time_data(
                config=self.config, file_keyword=self.config.test_year
            )
            return self.X_test, self.y_test, self.keys, self.shape
