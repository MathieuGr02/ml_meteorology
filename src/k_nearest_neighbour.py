from typing import override

import numpy as np
import numpy.typing as npt
from cuml.metrics import mean_absolute_error

from model import Config, Model
from utils import track_time


class KNearestNeighbourRegression(Model):
    dists: npt.NDArray[np.float64]
    indices: npt.NDArray[np.int64]

    def __init__(self, knr, neighbours: list[int], config: Config):
        super().__init__(config)
        self.knr = knr
        self.neighbours = neighbours

    def name(self) -> str:
        return "K Nearest Neighbour"

    def get_size(self) -> int:
        return self.knr.n_neighbors

    def get_neighbour_sizes(self) -> list[int]:
        return self.neighbours

    @override
    def train(self, X=None, y=None):
        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train
        self.train_time, _ = track_time(lambda: self.knr.fit(X, y))

    @override
    def predict(self, X=None):
        if X is None:
            X = self.X_test
        self.predict_time, (self.dists, self.indices) = track_time(
            lambda: self.knr.kneighbors(X)
        )

    @override
    def mae(self, key: str, y_test=None, output=None) -> list[float]:
        maes = []

        for k in self.neighbours:
            indices_k = self.indices[:, :k]
            self.outputs = np.mean(self.y_train[indices_k], axis=1)

            maes.append(super().mae(key, self.y_test, self.outputs))

        return maes
