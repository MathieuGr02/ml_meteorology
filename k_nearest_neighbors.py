from collections import defaultdict
from cuml.metrics import mean_absolute_error
from cuml.neighbors import KNeighborsRegressor
from aggregate import DataGroups, get_data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()


def train(knr, X, y) -> None:
    knr.fit(X, y)


def predict(knr, X, y) -> float:
    output = knr.predict(X)
    return mean_absolute_error(output, y)


def prepare_data(group: DataGroups):
    return get_data(group)


if __name__ == "__main__":
    groups = [
        DataGroups.SurfaceAirTemperatureA,
        DataGroups.OzoneA,
        DataGroups.RelativeHumiditySurfaceA,
    ]

    for group in groups:
        print("Getting data")

        sizes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

        results = defaultdict(list)

        training, training_target, test, test_target = prepare_data(group)

        for size in sizes:
            knr = KNeighborsRegressor(n_neighbors=size)
            train(knr, training, training_target)
            error = predict(knr, test, test_target)
            results[size].append(error)

        errors = [results[key] for key in results]
        print(list(results.keys())[np.argmin(errors)])

        plt.figure(figsize=(10, 5))
        sns.barplot(results, color="C0")
        plt.title(f"Optimal K-Nearest Neighbours model | {group.get_group()}")
        plt.ylabel(f"Mean Absolute Error (MAE) | {group.get_unit()}")
        plt.xlabel("Number of Neighbours")
        plt.tight_layout()
        plt.savefig(f"figures/knn_opt_{group.value}.png")
