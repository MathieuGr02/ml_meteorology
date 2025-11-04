from collections import defaultdict
from cuml.metrics import mean_absolute_error
from cuml.preprocessing import PolynomialFeatures as cuMLPolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures as sklearnPolynomialFeatures
from cuml.linear_model import LinearRegression as cuMLLinearRegression
from sklearn.linear_model import LinearRegression as sklearnLinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gc
import cupy as cp
import numpy.typing as npt

sns.set_theme()

from aggregate import DataGroups, get_data


def train(f, X, y) -> None:
    f.fit(X, y)


def predict(f, X, y) -> tuple[float, npt.NDArray[np.float64]]:
    output = f.predict(X)
    return mean_absolute_error(output, y), output


def degree_fit(X_train, X_test, p):
    training_p = p.fit_transform(X_train)
    test_p = p.transform(X_test)
    return training_p, test_p


def prepare_data(group: DataGroups):
    return get_data(group)


if __name__ == "__main__":
    groups = [
        # DataGroups.SurfaceAirTemperatureA,
        DataGroups.OzoneA
        # DataGroups.RelativeHumiditySurfaceA,
    ]

    for group in groups:
        print("Getting data")

        degrees = [1, 2, 3]

        results_degree = defaultdict(list)

        try:
            training, training_target, test, test_target = prepare_data(group)

            for degree in degrees:
                print(f"Degree: {degree}")

                regression = cuMLLinearRegression()

                p = cuMLPolynomialFeatures(degree)

                training_p, test_p = degree_fit(training, test, p)

                train(regression, training_p, training_target)
                error = predict(regression, test_p, test_target)

                results_degree[degree].append(error)

                del training_p, test_p, regression
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
        except Exception as E:
            print(f"{E}")
            """
            training, training_target, test, test_target = prepare_data(group)

            for degree in degrees:
                print(f"Degree: {degree}")

                regression = sklearnLinearRegression()

                p = sklearnPolynomialFeatures(degree)

                training_p, test_p = degree_fit(training, test, p)

                train(regression, training_p, training_target)
                error = predict(regression, test_p, test_target)

                results_degree[degree].append(error)
                """

        for key in results_degree.keys():
            results_degree[key] = np.average(results_degree[key])

        plt.figure(figsize=(10, 5))
        sns.barplot(results_degree)
        plt.title(f"Optimal linear regression model | {group.get_group()}")
        plt.ylabel(f"Mean Absolute Error (MAE) | {group.get_unit()}")
        plt.xlabel("Polynomial degree")
        plt.tight_layout()
        # plt.savefig(f"figures/lr_opt_{group.value}.png")
