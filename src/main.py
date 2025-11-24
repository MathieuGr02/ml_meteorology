import argparse
import itertools
import json
import os
import warnings
from collections import defaultdict
from enum import Enum
from typing import Self

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
from cuml.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn

from aggregate import get_data, get_time_data
from download import download_files
from gaussian_process import GaussianProcess
from k_nearest_neighbour import KNearestNeighbourRegression
from model import Config, Model
from neural_network import MLP, NeuralNetwork
from regression import Regression

warnings.simplefilter(action="ignore", category=FutureWarning)

import cuml

cuml.set_global_output_type("numpy")


def get_compare_models(
    config: Config, input: int, output: int
) -> tuple[NeuralNetwork, KNearestNeighbourRegression, NeuralNetwork, GaussianProcess]:
    models = []

    lr = nn.Linear(input, output)
    optimizer = torch.optim.SGD(lr.parameters(), lr=0.1)
    r = NeuralNetwork(
        network=lr,
        loss_function=nn.MSELoss(),
        config=config,
        optimizer=optimizer,
    )
    models.append(r)

    knr = KNeighborsRegressor(n_neighbors=10)
    k = KNearestNeighbourRegression(knr=knr, neighbours=[10], config=config)
    models.append(k)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input * 4),
            nn.ReLU(),
            nn.Linear(input * 4, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input),
            nn.ReLU(),
            nn.Linear(input, output),
        ),
        "N3",
    )
    n = NeuralNetwork(mlp, loss_function=nn.MSELoss(), config=config)
    models.append(n)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output)
    gp = GaussianProcess(
        likelihood=likelihood,
        n_inducing_points=50,
        config=config,
        input=input,
        output=output,
    )
    models.append(gp)

    return models


def get_search_models(
    config: Config, input: int, output: int
) -> tuple[
    list[Regression],
    list[KNearestNeighbourRegression],
    list[NeuralNetwork],
    list[GaussianProcess],
]:
    models = ()

    lr_models = []
    lr = nn.Linear(input, output)
    optimizer = torch.optim.SGD(lr.parameters(), lr=0.1)
    r = NeuralNetwork(
        network=lr,
        loss_function=nn.MSELoss(),
        config=config,
        optimizer=optimizer,
    )
    lr_models.append(r)
    models += (lr_models,)

    knr_models = []
    n_neighbours = [
        1,
        2,
        3,
        4,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        150,
        200,
    ]
    knr = KNearestNeighbourRegression(
        knr=KNeighborsRegressor(n_neighbors=max(n_neighbours)),
        neighbours=n_neighbours,
        config=config,
    )
    knr_models.append(knr)
    models += (knr_models,)

    mlp_models = []

    loss_functions = [nn.MSELoss(), nn.HuberLoss(), nn.L1Loss()]

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input),
            nn.ReLU(),
            nn.Linear(input, input),
            nn.ReLU(),
            nn.Linear(input, output),
        ),
        "N1",
    )
    mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input),
            nn.ReLU(),
            nn.Linear(input, output),
        ),
        "N2",
    )
    mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input * 4),
            nn.ReLU(),
            nn.Linear(input * 4, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input),
            nn.ReLU(),
            nn.Linear(input, output),
        ),
        "N3",
    )
    mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input * 4),
            nn.ReLU(),
            nn.Linear(input * 4, input * 8),
            nn.ReLU(),
            nn.Linear(input * 8, input * 4),
            nn.ReLU(),
            nn.Linear(input * 4, input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input),
            nn.ReLU(),
            nn.Linear(input, output),
        ),
        "N4",
    )
    mlp_models.append(mlp)

    nn_models = []
    for loss_function, network in itertools.product(loss_functions, mlp_models):
        network = NeuralNetwork(
            network=network,
            loss_function=loss_function,
            config=config,
        )

        nn_models.append(network)

    models += (nn_models,)

    inducing_points = [100, 250, 500, 750]

    gp_models = []
    for inducing_point in inducing_points:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output)
        gp = GaussianProcess(
            likelihood=likelihood,
            n_inducing_points=inducing_point,
            config=config,
            output=output,
            input=input,
        )
        gp_models.append(gp)
    models += (gp_models,)

    return models


class Action(Enum):
    Compare = "compare"
    Search = "search"
    Download = "download"
    PlotCompare = "plotcompare"
    PlotMap = "plotmap"

    def __str__(self):
        return self.value

    def __matches__(self, other):
        return self.value == other.value

    def is_search(self):
        return self.__matches__(Action.Search)

    def is_compare(self):
        return self.__matches__(Action.Compare)

    def is_download(self):
        return self.__matches__(Action.Download)

    def is_plot_compare(self):
        return self.__matches__(Action.PlotCompare)

    def is_plot_map(self):
        return self.__matches__(Action.PlotMap)


lead_max = 10


def search(
    config: Config,
    ignore_lr=False,
    ignore_knr=False,
    ignore_nn=False,
    ignore_gp=False,
):
    lr_output = pd.DataFrame(
        columns=[
            "lead",
            "key",
            "mae",
            "train_time",
            "predict_time",
        ]
    )
    knr_output = pd.DataFrame(
        columns=[
            "neighbours",
            "lead",
            "key",
            "mae",
            "train_time",
            "predict_time",
        ]
    )
    nn_output = pd.DataFrame(
        columns=[
            "network",
            "loss_function",
            "lead",
            "key",
            "mae",
            "train_time",
            "predict_time",
        ]
    )
    gp_output = pd.DataFrame(
        columns=[
            "inducing_points",
            "lead",
            "key",
            "mae",
            "train_time",
            "predict_time",
        ]
    )

    print(f"Running search for lead time {lead_max}")

    config.leads = lead_max

    print("Getting train data")
    X_train, y_train, keys, shape = get_time_data(
        config, file_keyword=config.train_year
    )
    print(X_train.shape)
    quit()
    print("Getting test data")
    X_test, y_test, *_ = get_time_data(config, file_keyword=config.test_year)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Keys: {keys} | Amount: {len(keys)}")

    lr_models, knr_models, nn_models, gp_models = get_search_models(
        config, X_train.shape[1], y_train.shape[1]
    )

    if not ignore_lr:
        for i, model in enumerate(lr_models):
            model.set_train_data(X_train, y_train, keys, shape)
            model.set_test_data(X_test, y_test, keys, shape)
            print(f"Running {model.name()} ({i}/{len(lr_models)})")
            model.run()

            for key in keys:
                print(f"Calculating MAE of {key} for {model.name()}")
                lr_output = lr_output._append(
                    {
                        "lead": lead,
                        "key": key,
                        "mae": model.mae(key),
                        "train_time": model.get_train_time(),
                        "predict_time": model.get_predict_time(),
                    },
                    ignore_index=True,
                )

            lr_output.to_csv("output/lr_metrics.csv", index=False)

    if not ignore_knr:
        for model in knr_models:
            model.set_train_data(X_train, y_train, keys, shape)
            model.set_test_data(X_test, y_test, keys, shape)
            print(f"Running {model.name()}")
            model.run()
            for key in keys:
                print(f"Calculating MAE of {key} for {model.name()}")
                for size, mae in zip(
                    model.get_neighbour_sizes(),
                    model.mae(key),
                ):
                    knr_output = knr_output._append(
                        {
                            "neighbours": size,
                            "lead": lead,
                            "key": key,
                            "mae": mae,
                            "train_time": model.get_train_time(),
                            "predict_time": model.get_predict_time(),
                        },
                        ignore_index=True,
                    )

            knr_output.to_csv("output/knr_metrics.csv", index=False)

    if not ignore_nn:
        for i, model in enumerate(nn_models):
            model.set_train_data(X_train, y_train, keys, shape)
            model.set_test_data(X_test, y_test, keys, shape)
            print(f"Running {model.name()} ({i}/{len(nn_models)})")
            model.run()

            for key in keys:
                print(f"Calculating MAE of {key} for {model.name()}")
                nn_output = nn_output._append(
                    {
                        "network": model.network_name(),
                        "loss_function": model.get_loss_function(),
                        "lead": lead,
                        "key": key,
                        "mae": model.mae(key),
                        "train_time": model.get_train_time(),
                        "predict_time": model.get_predict_time(),
                    },
                    ignore_index=True,
                )

            nn_output.to_csv("output/nn_metrics.csv", index=False)

    if not ignore_gp:
        for i, model in enumerate(gp_models):
            model.set_train_data(X_train, y_train, keys, shape)
            model.set_test_data(X_test, y_test, keys, shape)
            print(f"Running {model.name()} ({i}/{len(gp_models)})")
            model.run()

            for key in keys:
                print(f"Calculating MAE of {key} for {model.name()}")
                gp_output = gp_output._append(
                    {
                        "inducing_points": model.get_inducing_points(),
                        "lead": lead,
                        "key": key,
                        "mae": model.mae(key),
                        "train_time": model.get_train_time(),
                        "predict_time": model.get_predict_time(),
                    },
                    ignore_index=True,
                )

            gp_output.to_csv("output/gp_metrics.csv", index=False)


def compare(
    config: Config, ignore_lr=False, ignore_knr=False, ignore_nn=False, ignore_gp=False
):
    compare_metrics = pd.DataFrame(
        columns=["model", "lead", "key", "mae", "train_time", "predict_time"]
    )

    if os.path.exists("output/compare_metrics.csv"):
        compare_metrics = pd.read_csv("output/compare_metrics.csv")

    config.leads = lead_max

    print("Getting train data")
    X_train, y_train, keys, shape = get_time_data(
        config, file_keyword=config.train_year
    )
    print(config.leads, X_train.shape, y_train.shape)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)

    print(f"Keys: {keys} | Amount: {len(keys)}")

    lr, knr, network, gp = get_compare_models(
        config, X_train.shape[1], y_train.shape[1]
    )

    if not ignore_lr:
        print(f"Running Linear Regression")
        lr.set_train_data(X_train, y_train, keys, shape)
        trainloader = lr.trainloader()
        lr.train(trainloader)

        print("Getting test data")
        X_test, y_test, keys, shape = get_time_data(
            config, file_keyword=config.test_year
        )

        print(X_test.shape, y_test.shape)

        print("Scaling test data")
        X_test = X_scaler.transform(X_test)

        lr.set_test_data(X_test, y_test, keys, shape)

        testloader = lr.testloader()
        lr.predict(testloader)

        for key in keys:
            for lead in range(1, lead_max + 1):
                y_test_lead = y_test[:, len(keys) * (lead - 1) : len(keys) * lead]
                compare_metrics = compare_metrics._append(
                    {
                        "model": "Linear Regression",
                        "lead": lead,
                        "key": key,
                        "mae": lr.mae(key=key, y_test=y_test_lead),
                        "rmse": lr.rmse(key=key, y_test=y_test_lead),
                        "train_time": lr.get_train_time(),
                        "predict_time": lr.get_predict_time(),
                    },
                    ignore_index=True,
                )

        compare_metrics.to_csv("output/compare_metrics.csv", index=False)

    if not ignore_knr:
        print(f"Running K Nearest Neighbours")
        knr.set_train_data(X_train, y_train, keys, shape)
        knr.train(X_train, y_train)

        print("Getting test data")
        X_test, y_test, keys, shape = get_time_data(
            config, file_keyword=config.test_year
        )

        print("Scaling test data")
        X_test = X_scaler.transform(X_test)

        knr.set_test_data(X_test, y_test, keys, shape)

        knr.predict()

        for key in keys:
            for lead in range(1, lead_max + 1):
                y_test_lead = y_test[:, len(keys) * (lead - 1) : len(keys) * lead]
                compare_metrics = compare_metrics._append(
                    {
                        "model": "K Nearest Neighbours",
                        "lead": lead,
                        "key": key,
                        "mae": knr.mae(key=key, y_test=y_test_lead)[0],
                        "rmse": knr.rmse(key=key, y_test=y_test_lead),
                        "train_time": knr.get_train_time(),
                        "predict_time": knr.get_predict_time(),
                    },
                    ignore_index=True,
                )

        compare_metrics.to_csv("output/compare_metrics.csv", index=False)

    if not ignore_nn:
        print(f"Running Neural Network")
        network.set_train_data(X_train, y_train, keys, shape)
        trainloader = network.trainloader()
        network.train(trainloader)

        print("Getting test data")
        X_test, y_test, keys, shape = get_time_data(
            config, file_keyword=config.test_year
        )

        print("Scaling test data")
        X_test = X_scaler.transform(X_test)

        network.set_test_data(X_test, y_test, keys, shape)

        testloader = network.testloader()
        network.predict(testloader)

        for key in keys:
            for lead in range(1, lead_max + 1):
                y_test_lead = y_test[:, len(keys) * (lead - 1) : len(keys) * lead]
                compare_metrics = compare_metrics._append(
                    {
                        "model": "Neural Network",
                        "lead": lead,
                        "key": key,
                        "mae": network.mae(key=key, y_test=y_test_lead),
                        "rmse": network.rmse(key=key, y_test=y_test_lead),
                        "train_time": network.get_train_time(),
                        "predict_time": network.get_predict_time(),
                    },
                    ignore_index=True,
                )

        compare_metrics.to_csv("output/compare_metrics.csv", index=False)

    if not ignore_gp:
        print(f"Running Gaussian Process")
        gp.set_train_data(X_train, y_train, keys, shape)
        print("Trainloading")
        _, trainloader = gp.trainloader()
        print("Creating model")
        gp.create_model()
        print("Training")
        gp.train(trainloader)

        print("Getting test data")
        X_test, y_test, keys, shape = get_time_data(
            config, file_keyword=config.test_year
        )

        print("Scaling test data")
        X_test = X_scaler.transform(X_test)

        gp.set_test_data(X_test, y_test, keys, shape)

        testloader = gp.testloader()
        gp.predict(testloader)

        for key in keys:
            for lead in range(1, lead_max + 1):
                y_test_lead = y_test[:, len(keys) * (lead - 1) : len(keys) * lead]
                compare_metrics = compare_metrics._append(
                    {
                        "model": "Gaussian Process",
                        "lead": lead,
                        "key": key,
                        "mae": gp.mae(key=key, y_test=y_test_lead),
                        "rmse": gp.rmse(key=key, y_test=y_test_lead),
                        "train_time": gp.get_train_time(),
                        "predict_time": gp.get_predict_time(),
                    },
                    ignore_index=True,
                )

    compare_metrics.to_csv("output/compare_metrics.csv", index=False)


def read_models():
    def get_best_model(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        df_mean_mae = (
            df.groupby(keys + ["key"]).agg(mean_mae=("mae", "mean")).reset_index()
        )
        df_best_per_key_idx = df_mean_mae.groupby("key")["mean_mae"].idxmin()
        df_best_models_per_key = df_mean_mae.loc[df_best_per_key_idx].reset_index(
            drop=True
        )
        return df_best_models_per_key

    lr_metrics = pd.read_csv("output/lr_metrics.csv")
    knr_metrics = pd.read_csv("output/knr_metrics.csv")
    nn_metrics = pd.read_csv("output/nn_metrics.csv")
    gp_metrics = pd.read_csv("output/gp_metrics.csv")

    lr_best_models_per_key = get_best_model(lr_metrics, [])
    print("Linear Regression:\n", lr_best_models_per_key)

    knr_best_models_per_key = get_best_model(knr_metrics, ["neighbours"])
    print("K Nearest Neighbours Regression:\n", knr_best_models_per_key)
    counts = knr_best_models_per_key["neighbours"].value_counts()
    optimal_neighbours = counts.idxmax()
    print(f"Optimal Neighbours size: {optimal_neighbours}")

    nn_best_models_per_key = get_best_model(nn_metrics, ["network", "loss_function"])
    print("Neural Network:\n", nn_best_models_per_key)

    nn_best_models_per_key["comb"] = (
        nn_best_models_per_key["network"]
        + "_"
        + nn_best_models_per_key["loss_function"]
    )

    counts = nn_best_models_per_key["comb"].value_counts()
    optimal_combination = counts.idxmax()

    print("Optimal Neural Network combination: ", optimal_combination)

    gp_best_models_per_key = get_best_model(gp_metrics, ["inducing_points"])
    print("Gaussian Process:\n", gp_best_models_per_key)
    counts = gp_best_models_per_key["inducing_points"].value_counts()
    optimal_inducing_points = counts.idxmax()
    print(f"Gaussian Process optimal inducing points: {optimal_inducing_points}")


def plot_compare(main_key: str, values: list[str], unit: str, n, m):
    compare_metrics = pd.read_csv("output/compare_metrics.csv")

    fig, axs = plt.subplots(n, m, figsize=(20, 10))

    keys = [f"{main_key}-{value}" for value in values]

    k, l = 0, 0
    for i, (key, value) in enumerate(zip(keys, values)):
        k = i % m
        if k == 0 and i != 0:
            l += 1

        ax = axs[l, k]

        sns.lineplot(
            data=compare_metrics[compare_metrics["key"] == key],
            x="lead",
            y="mae",
            hue="model",
            ax=ax,
        )
        ax.set_title(f"{value} {unit}")
        ax.set_xlabel("Lead time")
        ax.set_ylabel(f"MAE {unit}")
        ax.set_xticks(range(1, 11))

    handles, labels = ax.get_legend_handles_labels()

    n, m = axs.shape
    for i in range(n):
        for j in range(m):
            print(axs[i, j])
            axs[i, j].legend().remove()

    fig.suptitle(f"Model Comparison for {main_key}")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    plt.savefig(f"figures/compare_models_mae_{main_key}.png")

    fig, axs = plt.subplots(n, m, figsize=(20, 10))
    k, l = 0, 0
    for i, (key, value) in enumerate(zip(keys, values)):
        k = i % m
        if k == 0 and i != 0:
            l += 1

        ax = axs[l, k]

        plt.legend()
        sns.lineplot(
            data=compare_metrics[compare_metrics["key"] == key],
            x="lead",
            y="rmse",
            hue="model",
            ax=ax,
        )
        ax.set_title(f"{value} {unit}")
        ax.set_xlabel("Lead time")
        ax.set_ylabel(f"RMSE")
        ax.set_xticks(range(1, 11))

    n, m = axs.shape
    for i in range(n):
        for j in range(m):
            print(axs[i, j])
            axs[i, j].legend().remove()

    fig.suptitle(f"Model Comparison for {main_key}")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    plt.savefig(f"figures/compare_models_rmse_{main_key}.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=compare_metrics, x="model", y="train_time", ax=ax)
    ax.set_title("Model training time comparison")
    ax.set_ylabel("Train time (s)")
    plt.savefig(f"figures/compare_models_train_time.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=compare_metrics, x="model", y="predict_time", ax=ax)
    ax.set_title("Model prediction time comparison")
    ax.set_ylabel("Prediction time (s)")
    plt.savefig(f"figures/compare_models_predict_time.png")


def plot_map(config):
    file = os.listdir("./data")[0]
    dataset = xr.open_dataset(f"./data/{file}")
    lon_low, lon_high = config.lon_low + 180, config.lon_high + 180
    lat_low, lat_high = (
        90 - config.lat_high,
        90 - config.lat_low,
    )

    area = dataset["Topography"][lat_low:lat_high, lon_low:lon_high]
    print(area)
    # area = np.flipud(area.to_numpy())
    print(area.shape)
    print(area[0, 0])
    plt.imshow(area, cmap="magma", origin="upper", extent=[-20, 59, 31, 80])
    plt.grid(None)
    plt.xlabel("° Longitude")
    plt.ylabel("° Latitude")
    plt.tight_layout()
    plt.savefig("figures/map.png")


class ModelNames(Enum):
    LR = "lr"
    KNR = "knr"
    NN = "nn"
    GP = "gp"

    def __str__(self) -> str:
        return self.value


if __name__ == "__main__":
    print("Starting")
    parser = argparse.ArgumentParser(
        prog="ML Meteorology",
        description="Machine Learning for weather forecast",
    )

    parser.add_argument("-t", "--type", type=Action, choices=list(Action))
    parser.add_argument(
        "-i", "--ignore", nargs="+", type=ModelNames, choices=list(ModelNames)
    )
    args = parser.parse_args()

    if args.type is not None:
        action: Action = args.type

        if action.is_download():
            print("Downloading files")
            download_files()
            exit(0)

        # Europe
        lon_low = -20
        lon_high = 60
        lat_low = 30
        lat_high = 80

        config = Config(lon_low, lon_high, lat_low, lat_high)

        if args.ignore is not None:
            ignore_lr = ModelNames.LR in args.ignore
            ignore_knr = ModelNames.KNR in args.ignore
            ignore_nn = ModelNames.NN in args.ignore
            ignore_gp = ModelNames.GP in args.ignore
        else:
            ignore_lr = False
            ignore_knr = False
            ignore_nn = False
            ignore_gp = False

        print("Ignore LR: ", ignore_lr)
        print("Ignore KNR: ", ignore_knr)
        print("Ignore NN: ", ignore_nn)
        print("Ignore GP: ", ignore_gp)

        if action.is_plot_map():
            plot_map(config)

        if action.is_search():
            print("Search")
            search(
                config,
                ignore_lr=ignore_lr,
                ignore_nn=ignore_nn,
                ignore_knr=ignore_knr,
                ignore_gp=ignore_gp,
            )

        if action.is_compare():
            print("Compare")
            compare(
                config,
                ignore_lr=ignore_lr,
                ignore_nn=ignore_nn,
                ignore_knr=ignore_knr,
                ignore_gp=ignore_gp,
            )

        if action.is_plot_compare():
            print("Plot Compare")
            main_key = "Temperature"
            unit = "K"
            values = [
                "StdPressureLev-1.0",
                "StdPressureLev-1.5",
                "StdPressureLev-2.0",
                "StdPressureLev-3.0",
                "StdPressureLev-5.0",
                "StdPressureLev-7.0",
                "StdPressureLev-10.0",
                "StdPressureLev-15.0",
                "StdPressureLev-20.0",
                "StdPressureLev-30.0",
                "StdPressureLev-50.0",
                "StdPressureLev-70.0",
                "StdPressureLev-100.0",
                "StdPressureLev-150.0",
                "StdPressureLev-200.0",
                "StdPressureLev-250.0",
                "StdPressureLev-300.0",
                "StdPressureLev-400.0",
                "StdPressureLev-500.0",
                "StdPressureLev-600.0",
                "StdPressureLev-700.0",
                "StdPressureLev-850.0",
                "StdPressureLev-925.0",
                "StdPressureLev-1000.0",
            ]
            plot_compare(main_key, values, unit, 4, 6)

            main_key = "RelHum"
            unit = "hPa"
            values = [
                "H2OPressureLev-150.0",
                "H2OPressureLev-200.0",
                "H2OPressureLev-250.0",
                "H2OPressureLev-300.0",
                "H2OPressureLev-400.0",
                "H2OPressureLev-500.0",
                "H2OPressureLev-600.0",
                "H2OPressureLev-700.0",
                "H2OPressureLev-850.0",
                "H2OPressureLev-925.0",
                "H2OPressureLev-1000.0",
            ]
            plot_compare(main_key, values, unit, 2, 6)
    else:
        print("Please provide --type")
        exit(1)
