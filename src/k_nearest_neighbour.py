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

    def run(self):
        self.X_train, self.y_train, *_ = super().train_data()
        self.X_test, self.y_test, *_ = super().test_data()

        print(f"Running {self.name()} fit")
        self.train_time, _ = track_time(
            lambda: self.knr.fit(self.X_train, self.y_train)
        )

        self.dists, self.indices = self.knr.kneighbors(self.X_test)

        print(f"Running {self.name()} predict")
        self.predict_time, self.output = track_time(
            lambda: self.knr.predict(self.X_test)
        )

    @override
    def mae(self, key: str, y_test=None, output=None) -> list[float]:
        print(super().mae(key, self.y_test, self.output))

        maes = []

        for k in self.neighbours:
            indices_k = self.indices[:, :k]
            y_pred = np.mean(self.y_train[indices_k], axis=1)

            maes.append(super().mae(key, self.y_test, y_pred))

        return maes
