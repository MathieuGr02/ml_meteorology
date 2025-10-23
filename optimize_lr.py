from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from cuml.metrics import mean_absolute_error
from cuml.preprocessing import PolynomialFeatures
from cuml.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aggregate import DataGroups, get_data


def lr(degree, training, training_target, test, test_target) -> float:
    print(f"Degree: {degree}")
    p = PolynomialFeatures(degree)

    training = p.fit_transform(training)
    test = p.fit_transform(test)

    lr = LinearRegression()

    lr.fit(training, training_target)

    prediction = lr.predict(test)

    error = mean_absolute_error(prediction, test_target)

    return error


if __name__ == "__main__":
    group = DataGroups.SurfaceAirTemperatureA

    print("Getting data")
    training, training_target, test, test_target = get_data(group)

    degrees = [1, 2, 3]

    results_degree = defaultdict(list)

    for degree in degrees:
        error = lr(degree, training, training_target, test, test_target)
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
