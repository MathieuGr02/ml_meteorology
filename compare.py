from cuml.neighbors import KNeighborsRegressor
from cuml.preprocessing import PolynomialFeatures
import torch
import seaborn as sns
from torch import nn
import matplotlib.pyplot as plt
import neural_network
import gaussian_process_old
import k_nearest_neighbors
import regression_old
import gpytorch
import time
from typing import Any
import pandas as pd
import numpy as np

sns.set_theme()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from aggregate import DataGroups

    group_A = DataGroups.SurfaceAirTemperatureA
    group_D = DataGroups.SurfaceAirTemperatureD

    print(f"Performing comparison for {group_A} and {group_D}")

    predict_times_A = {}
    predict_times_D = {}
    train_times_A = {}
    train_times_D = {}
    errors_A = {}
    errors_D = {}

    # Neural Network
    nn_trainloader_A, nn_testloader_A = neural_network.prepare_data(group_A)
    nn_trainloader_D, nn_testloader_D = neural_network.prepare_data(group_D)

    print("Neural Network")
    features_A = nn_trainloader_A.dataset.get_features()
    features_D = nn_trainloader_A.dataset.get_features()

    mlp_A, epoch_A, lr_A, loss_function_A = group_A.get_neural_network(features_A)
    mlp_D, epoch_D, lr_D, loss_function_D = group_D.get_neural_network(features_D)

    nn_train_time_A, _ = track_time(
        lambda: neural_network.train(
            mlp_A, nn_trainloader_A, loss_function_A, epoch=epoch_A, lr=lr_A
        )
    )
    nn_train_time_D, _ = track_time(
        lambda: neural_network.train(
            mlp_D, nn_trainloader_D, loss_function_D, epoch=epoch_D, lr=lr_D
        )
    )

    nn_predict_time_A, nn_error_A = track_time(
        lambda: neural_network.predict(mlp_A, nn_testloader_A)
    )
    nn_predict_time_D, nn_error_D = track_time(
        lambda: neural_network.predict(mlp_D, nn_testloader_D)
    )

    errors_A["Neural Network"] = nn_error_A
    errors_D["Neural Network"] = nn_error_D
    train_times_A["Neural Network"] = nn_train_time_A
    train_times_D["Neural Network"] = nn_train_time_D
    predict_times_A["Neural Network"] = nn_predict_time_A
    predict_times_D["Neural Network"] = nn_predict_time_D

    print(nn_error_A, nn_error_D)

    # Linear Regression
    print("Linear Regression")
    lr_training_A, lr_training_target_A, lr_test_A, lr_test_target_A = (
        regression_old.prepare_data(group_A)
    )
    lr_training_D, lr_training_target_D, lr_test_D, lr_test_target_D = (
        regression_old.prepare_data(group_D)
    )

    lr_A, p_A = group_A.get_linear_regression()
    lr_D, p_D = group_D.get_linear_regression()

    training_p_A, test_p_A = regression_old.degree_fit(lr_training_A, lr_test_A, p_A)
    training_p_D, test_p_D = regression_old.degree_fit(lr_training_D, lr_test_D, p_D)

    lr_train_time_A, _ = track_time(
        lambda: regression_old.train(lr_A, training_p_A, lr_training_target_A)
    )
    lr_train_time_D, _ = track_time(
        lambda: regression_old.train(lr_D, training_p_D, lr_training_target_D)
    )

    lr_predict_time_A, lr_error_A = track_time(
        lambda: regression_old.predict(lr_A, test_p_A, lr_test_target_A)
    )
    lr_predict_time_D, lr_error_D = track_time(
        lambda: regression_old.predict(lr_D, test_p_D, lr_test_target_D)
    )

    errors_A["Linear Regression"] = lr_error_A
    errors_D["Linear Regression"] = lr_error_D
    train_times_A["Linear Regression"] = lr_train_time_A
    train_times_D["Linear Regression"] = lr_train_time_D
    predict_times_A["Linear Regression"] = lr_predict_time_A
    predict_times_D["Linear Regression"] = lr_predict_time_D

    print(lr_error_A, lr_error_D)

    # K Nearest Neighbors
    print("K-Nearest Neighbors")
    knr_training_A, knr_training_target_A, knr_test_A, knr_test_target_A = (
        k_nearest_neighbors.prepare_data(group_A)
    )
    knr_training_D, knr_training_target_D, knr_test_D, knr_test_target_D = (
        k_nearest_neighbors.prepare_data(group_D)
    )

    knr_A = group_A.get_k_nearest_neighbours()
    knr_D = group_D.get_k_nearest_neighbours()

    knr_train_time_A, _ = track_time(
        lambda: k_nearest_neighbors.train(knr_A, knr_training_A, knr_training_target_A)
    )
    knr_train_time_D, _ = track_time(
        lambda: k_nearest_neighbors.train(knr_D, knr_training_D, knr_training_target_D)
    )

    knr_predict_time_A, knr_error_A = track_time(
        lambda: k_nearest_neighbors.predict(knr_A, knr_test_A, knr_test_target_A)
    )
    knr_predict_time_D, knr_error_D = track_time(
        lambda: k_nearest_neighbors.predict(knr_D, knr_test_D, knr_test_target_D)
    )

    errors_A["K-Nearest Neighbors"] = knr_error_A
    errors_D["K-Nearest Neighbors"] = knr_error_D
    train_times_A["K-Nearest Neighbors"] = knr_train_time_A
    train_times_D["K-Nearest Neighbors"] = knr_train_time_D
    predict_times_A["K-Nearest Neighbors"] = knr_predict_time_A
    predict_times_D["K-Nearest Neighbors"] = knr_predict_time_D

    print(knr_error_A, knr_error_D)

    # Gaussian Process
    print("Gaussian Process")

    epoch_A, lr_A, amount_inducing_points_A = group_A.get_gaussian_process()
    epoch_D, lr_D, amount_inducing_points_D = group_D.get_gaussian_process()

    gp_inducing_points_A, gp_trainloader_A, gp_testloader_A = (
        gaussian_process_old.prepare_data(group_A, amount_inducing_points_A)
    )
    gp_inducing_points_D, gp_trainloader_D, gp_testloader_D = (
        gaussian_process_old.prepare_data(group_D, amount_inducing_points_D)
    )

    gp_A = gaussian_process_old.GP(gp_inducing_points_A)
    gp_D = gaussian_process_old.GP(gp_inducing_points_D)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    gp_train_time_A, _ = track_time(
        lambda: gaussian_process_old.train(
            gp_A, likelihood, gp_trainloader_A, epoch=epoch_A, lr=lr_A
        )
    )
    gp_train_time_D, _ = track_time(
        lambda: gaussian_process_old.train(
            gp_D, likelihood, gp_trainloader_D, epoch=epoch_D, lr=lr_D
        )
    )
    gp_predict_time_A, gp_error_A = track_time(
        lambda: gaussian_process_old.predict(gp_A, likelihood, gp_testloader_A)
    )
    gp_predict_time_D, gp_error_D = track_time(
        lambda: gaussian_process_old.predict(gp_D, likelihood, gp_testloader_D)
    )

    errors_A["Gaussian Process"] = gp_error_A
    errors_D["Gaussian Process"] = gp_error_D
    train_times_A["Gaussian Process"] = gp_train_time_A
    train_times_D["Gaussian Process"] = gp_train_time_D
    predict_times_A["Gaussian Process"] = gp_predict_time_A
    predict_times_D["Gaussian Process"] = gp_predict_time_D

    print(gp_error_A, gp_error_D)

    df = pd.DataFrame.from_dict(
        {
            "method": list(errors_A.keys()) * 2,
            "type": ["A"] * len(errors_A) + ["D"] * len(errors_D),
            "error": [value for key, value in errors_A.items()]
            + [value for key, value in errors_D.items()],
            "train": [value for key, value in train_times_A.items()]
            + [value for key, value in train_times_D.items()],
            "predict": [value for key, value in predict_times_A.items()]
            + [value for key, value in predict_times_D.items()],
        }
    )

    print(df)

    plt.figure(figsize=(10, 5))
    sns.barplot(df, x="method", y="error", hue="type")
    plt.title(f"Methods error comparison | {group_A.get_group()}")
    plt.ylabel(f"Mean Absolute Error (MAE) | {group_A.get_unit()}")
    plt.xlabel("Methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/comparison_error_{group_A.value}_{group_D.value}.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(df, x="method", y="train", hue="type")
    plt.title(f"Methods training time comparison | {group_A.get_group()}")
    plt.ylabel(f"Training time (s)")
    plt.xlabel("Methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/comparison_train_time_{group_A.value}_{group_D.value}.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(df, x="method", y="predict", hue="type")
    plt.title(f"Methods prediction time comparison | {group_A.get_group()}")
    plt.ylabel(f"Prediction time (s)")
    plt.xlabel("Methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"figures/comparison_prediction_time_{group_A.value}_{group_D.value}.png"
    )
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(predict_times)
    plt.title("Methods predict time comparison for group 1")
    plt.ylabel("Prediction time (s)")
    plt.xlabel("Methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/comparison_predict_time_{group}.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(train_times)
    plt.title("Methods train time comparison for group 1")
    plt.ylabel("Train time (s)")
    plt.xlabel("Methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/comparison_train_time_{group}.png")
    """
