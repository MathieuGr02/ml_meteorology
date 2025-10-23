from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# from sklearn.metrics import mean_absolute_error
# from sklearn.neighbors import KNeighborsRegressor
from cuml.metrics import mean_absolute_error
from cuml.neighbors import KNeighborsRegressor
from aggregate import DataGroups, get_data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def knn(size, training, training_target, test, test_target) -> tuple[float, float]:
    print(f"Size: {size}")
    knr_uniform = KNeighborsRegressor(n_neighbors=size, weights="uniform")
    # knr_distance = KNeighborsRegressor(n_neighbors=size, weights="distance")

    knr_uniform.fit(training, training_target)
    # knr_distance.fit(training, training_target)

    prediction_uniform = knr_uniform.predict(test)
    # prediction_distance = knr_distance.predict(test)

    error_uniform = mean_absolute_error(prediction_uniform, test_target)
    error_distance = 0  # mean_absolute_error(prediction_distance, test_target)
    print(f"Size: {size} finished")
    return error_uniform, error_distance


if __name__ == "__main__":
    group = DataGroups.SurfaceAirTemperatureA

    print("Getting data")
    training, training_target, test, test_target = get_data(group)

    print(f"Training shape: {training.shape} | Test shape: {test.shape}")

    sizes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]

    results_uniform = defaultdict(list)
    results_distance = defaultdict(list)

    for size in sizes:
        error_uniform, error_distance = knn(
            size, training, training_target, test, test_target
        )
        results_uniform[size].append(error_uniform)
        results_distance[size].append(error_distance)

    for key in results_uniform.keys():
        results_uniform[key] = np.average(results_uniform[key])
        results_distance[key] = np.average(results_distance[key])

    df_uniform = pd.DataFrame(
        {
            "Category": list(results_uniform.keys()),
            "Value": list(results_uniform.values()),
            "Method": "Uniform",
        }
    )

    df_distance = pd.DataFrame(
        {
            "Category": list(results_distance.keys()),
            "Value": list(results_distance.values()),
            "Method": "Distance",
        }
    )

    df = pd.concat([df_uniform, df_distance], ignore_index=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Category", y="Value", hue="Method", data=df)
    plt.title("Uniform vs Distance Results")
    plt.ylabel("MAE")
    plt.xlabel("Number of Neighbors")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(f"figures/knn_opt_{group}.png")
    plt.show()
