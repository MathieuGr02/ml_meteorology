from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aggregate import DataGroups, get_data


def nn(
    layer, training_i, training_target_i, test_i, test_target_i
) -> tuple[int, float]:
    print(f"LAYER: {layer}")
    mlp = MLPRegressor(hidden_layer_sizes=layer)

    mlp.fit(training_i, training_target_i)

    prediction = mlp.predict(test_i)

    error = mean_absolute_error(prediction, test_target_i)

    return layer, error


if __name__ == "__main__":
    group = DataGroups.SurfaceAirTemperatureA

    print("Getting data")
    training, training_target, test, test_target = get_data(group)

    layers = [(10, 10), (10, 10, 10), (20, 20), (20, 20, 20), (30, 30, 30)]

    results_layers = defaultdict(list)

    print("Creating pool")
    with ThreadPoolExecutor(6) as executor:
        futures = []

        for iter, training_i, training_target_i, test_i, test_target_i in zip(
            range(len(training)), training, training_target, test, test_target
        ):
            for layer in layers:
                futures.append(
                    executor.submit(
                        nn, layer, training_i, training_target_i, test_i, test_target_i
                    )
                )

        for future in futures:
            layer, error = future.result()
            results_layers[layer.__str__()].append(error)

    for key in results_layers.keys():
        results_layers[key.__str__()] = np.average(results_layers[key.__str__()])

    plt.figure(figsize=(10, 5))
    sns.barplot(results_layers)
    plt.title("Neural Network Hidden Layer")
    plt.ylabel("MAE")
    plt.xlabel("Layers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/nn_opt_{group}.png")
    plt.show()
