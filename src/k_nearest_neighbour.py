from typing import override

import numpy as np
import numpy.typing as npt

from model import Config, Model
from resource_tracker import ResourceTracker


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
    def train(self, X, y):
        with ResourceTracker() as rt:
            self.knr.fit(X, y)
        self.train_resource = rt.results()

    @override
    def predict(self, X):
        with ResourceTracker() as rt:
            dists, indices = self.knr.kneighbors(X)
        self.predict_resource = rt.results()
        return dists, indices

    def predict_with_size(self, size: int, indices, y_train):
        indices_k = indices[:, :size]
        return np.mean(y_train[indices_k], axis=1)
