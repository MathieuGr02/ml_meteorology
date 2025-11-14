from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Config:
    lon_low: int
    lon_high: int
    lat_low: int
    lat_high: int
    X_train: npt.NDArray[np.float64] | None = None
    y_train: npt.NDArray[np.float64] | None = None
    X_test: npt.NDArray[np.float64] | None = None
    y_test: npt.NDArray[np.float64] | None = None
    keys: list[str] | None = None
    lag: int = 14  # Amount of steps used to train, 7 days in 12 h steps
    leads: int = 1  # Amount of steps ahead to predict
    train_year = "2015"
    test_year = "2016"
