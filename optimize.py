from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from aggregate import DataGroups, get_data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    training, training_target, test, test_target = get_data(
        DataGroups.SurfaceAirTemperatureA
    )

    print(training)

    results_uniform = defaultdict(list)
    results_distance = defaultdict(list)

    for iter, training_i, training_target_i, test_i, test_target_i in zip(
        range(len(training)), training, training_target, test, test_target
    ):
        print(f"Iteration: {iter}")

        for size in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]:
            print(f"Size: {size}")
            knr_uniform = KNeighborsRegressor(
                n_neighbors=size, weights="uniform", n_jobs=-1
            )
            knr_uniform.fit(training_i, training_target_i)

            prediction = knr_uniform.predict(test_i)

            error = mean_absolute_error(prediction, test_target_i)

            results_uniform[size].append(error)

            knr_distance = KNeighborsRegressor(
                n_neighbors=size, weights="distance", n_jobs=-1
            )
            knr_distance.fit(training_i, training_target_i)

            prediction = knr_distance.predict(test_i)

            error = mean_absolute_error(prediction, test_target_i)

            results_distance[size].append(error)

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
    plt.savefig("knn_opt.png")
    plt.show()
