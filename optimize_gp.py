from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aggregate import DataGroups, get_data


def gp(
    degree, training_i, training_target_i, test_i, test_target_i
) -> tuple[int, float]:
    p = PolynomialFeatures(degree)

    training_i = p.fit_transform(training_i)
    test_i = p.fit_transform(test_i)

    lr = LinearRegression()

    lr.fit(training_i, training_target_i)

    prediction = lr.predict(test_i)

    error = mean_absolute_error(prediction, test_target_i)

    return degree, error


if __name__ == "__main__":
    group = DataGroups.SurfaceAirTemperatureD

    print("Getting data")
    training, training_target, test, test_target = get_data(group)

    degrees = [1, 2, 3, 4, 5]

    results_degree = defaultdict(list)

    print("Creating pool")
    with ThreadPoolExecutor(8) as executor:
        futures = []

        for iter, training_i, training_target_i, test_i, test_target_i in zip(
            range(len(training)), training, training_target, test, test_target
        ):
            for degree in degrees:
                futures.append(
                    executor.submit(
                        lr, degree, training_i, training_target_i, test_i, test_target_i
                    )
                )

        for future in futures:
            degree, error = future.result()
            results_degree[degree].append(error)

    for key in results_degree.keys():
        results_degree[key] = np.average(results_degree[key])

    plt.figure(figsize=(10, 5))
    sns.barplot(results_degree)
    plt.title("Linear Regression Polynomial degree")
    plt.ylabel("MAE")
    plt.xlabel("Polynomial degree")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/lr_opt_{group}.png")
