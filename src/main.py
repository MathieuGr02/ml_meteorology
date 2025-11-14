import itertools
import json
import os
from collections import defaultdict
from enum import Enum

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cuml import MBSGDClassifier, MBSGDRegressor
from cuml.linear_model import LinearRegression
from cuml.neighbors import KNeighborsRegressor
from torch import nn

from aggregate import get_time_data
from download import download_files
from gaussian_process import GaussianProcess
from k_nearest_neighbour import KNearestNeighbourRegression
from model import Config, Model
from neural_network import MLP, NeuralNetwork
from regression import Regression

lead_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def get_compare_models(config: Config) -> list[Model]:
    models = []

    lr = LinearRegression()
    r = Regression(lr=lr, degree=1, config=config)
    models.append(r)

    knr = KNeighborsRegressor(n_neighbors=10)
    k = KNearestNeighbourRegression(knr=knr, config=config)
    models.append(k)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=15)
    gp = GaussianProcess(
        likelihood=likelihood,
        lr=0.01,
        epoch=1,
        n_inducing_points=100,
        config=config,
    )
    models.append(gp)

    features = 15
    network = MLP(
        nn.Sequential(
            nn.Linear(features * 6, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features * 4),
            nn.ReLU(),
            nn.Linear(features * 4, features * 8),
            nn.ReLU(),
            nn.Linear(features * 8, features * 4),
            nn.ReLU(),
            nn.Linear(features * 4, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features),
        ),
        "N4",
    )
    n = NeuralNetwork(
        network, loss_function=nn.MSELoss(), lr=0.01, epoch=1, config=config
    )
    models.append(n)

    return models


def get_search_models(
    config: Config,
) -> tuple[
    list[Regression],
    list[KNearestNeighbourRegression],
    list[NeuralNetwork],
    list[GaussianProcess],
]:
    # TODO: GET FEATURES
    features = 1

    models = ()

    lr_models = []
    for degree in [1, 2, 3, 4]:
        lr = MBSGDRegressor(loss="squared_loss", penalty=None, batch_size=512)
        r = Regression(lr=lr, degree=degree, config=config)
        lr_models.append(r)
    models += (lr_models,)

    knr_models = []
    n_neighbours = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    knr = KNearestNeighbourRegression(
        knr=KNeighborsRegressor(n_neighbors=max(n_neighbours)),
        neighbours=n_neighbours,
        config=config,
    )
    knr_models.append(knr)
    models += (knr_models,)

    mlp_models = []

    loss_functions = [nn.MSELoss(), nn.HuberLoss(), nn.L1Loss()]
    learning_rates = [0.1, 0.01, 0.001]
    epochs = [10, 20, 30]

    mlp = MLP(
        nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 1),
        ),
        "N1",
    )
    mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(features, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features),
            nn.ReLU(),
            nn.Linear(features, 1),
        ),
        "N2",
    )
    mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(features, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features * 4),
            nn.ReLU(),
            nn.Linear(features * 4, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features),
            nn.ReLU(),
            nn.Linear(features, 1),
        ),
        "N3",
    )
    mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(features, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features * 4),
            nn.ReLU(),
            nn.Linear(features * 4, features * 8),
            nn.ReLU(),
            nn.Linear(features * 8, features * 4),
            nn.ReLU(),
            nn.Linear(features * 4, features * 2),
            nn.ReLU(),
            nn.Linear(features * 2, features),
            nn.ReLU(),
            nn.Linear(features, 1),
        ),
        "N4",
    )
    mlp_models.append(mlp)

    nn_models = []
    for learning_rate, epoch, loss_function, network in itertools.product(
        learning_rates, epochs, loss_functions, mlp_models
    ):
        network = NeuralNetwork(
            network=network,
            lr=learning_rate,
            epoch=epoch,
            loss_function=loss_function,
            config=config,
        )

        nn_models.append(network)

    models += (nn_models,)

    gp_models = []
    for inducing_points in [100, 250, 500, 750, 1000]:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=15)
        gp = GaussianProcess(
            likelihood=likelihood,
            lr=0.01,
            epoch=1,
            n_inducing_points=inducing_points,
            config=config,
        )
        gp_models.append(gp)
    models += (gp_models,)

    return models


class Action(Enum):
    Compare = 0
    Search = 1
    Download = 2
    PlotCompare = 3
    PlotSearch = 4

    def is_search(self):
        match self:
            case Action.Search:
                return True
            case _:
                return False

    def is_compare(self):
        match self:
            case Action.Compare:
                return True
            case _:
                return False

    def is_download(self):
        match self:
            case Action.Download:
                return True
            case _:
                return False

    def is_plot_search(self):
        match self:
            case Action.PlotSearch:
                return True
            case _:
                return False


def search(
    config: Config,
    keys: str | list[str],
    ignore_lr=False,
    ignore_knr=False,
    ignore_nn=False,
    ignore_gp=False,
):
    lr_errors = {}
    knr_errors = {}
    nn_errors = {}
    gp_errors = {}

    # Initialize dictionaries for each key s.t. { "<key>": { "specific model - 1": [<errors>], ..., "specific model - n": [<errors>] } }
    for key in keys:
        lr_errors[key] = defaultdict(list)
        knr_errors[key] = defaultdict(list)
        nn_errors[key] = defaultdict(list)
        gp_errors[key] = defaultdict(list)

    if type(keys) is str:
        keys = [keys]

    for lead in lead_times:
        print(f"Running search for lead time {lead}")

        config.leads = lead

        X_train, y_train, keys, shape = get_time_data(
            config, file_keyword=config.train_year
        )
        X_test, y_test, *_ = get_time_data(config, file_keyword=config.test_year)

        config.X_train = X_train
        config.X_test = X_test
        config.y_train = y_train
        config.y_test = y_test
        config.keys = keys

        lr_models, knr_models, nn_models, gp_models = get_search_models(config)

        if not ignore_lr:
            for model in lr_models:
                model.run()

                # TODO: LR MODEL NAME DISTINCTION
                for key in keys:
                    lr_errors[key][""].append(model.mae(key))

        if not ignore_knr:
            for knr in knr_models:
                knr.run()
                for key in keys:
                    for size, error in zip(
                        knr.get_neighbour_sizes(),
                        knr.mae(key),
                    ):
                        knr_errors[key][size].append(error)

        if not ignore_nn:
            for model in nn_models:
                model.run()

                for key in keys:
                    nn_errors[key][model.network_name()].append(model.mae(key))

        if not ignore_gp:
            for model in gp_models:
                model.run()

                # TODO: GP MODEL NAME DISTINCTION
                for key in keys:
                    gp_errors[key][""].append(model.mae(key))

    if not ignore_lr:
        with open("outputs/lr_errors.json", "w") as f:
            json_string = json.dumps(lr_errors, indent=4)
            f.write(json_string)

    if not ignore_knr:
        with open("outputs/knr_errors.json", "w") as f:
            json_string = json.dumps(knr_errors, indent=4)
            f.write(json_string)

    if not ignore_nn:
        with open("outputs/nn_errors.json", "w") as f:
            json_string = json.dumps(nn_errors, indent=4)
            f.write(json_string)

    if not ignore_gp:
        with open("outputs/gp_errors.json", "w") as f:
            json_string = json.dumps(gp_errors, indent=4)
            f.write(json_string)


def compare(config: Config):
    lead_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lr, knr, mlp, gp = get_compare_models(config)


def plot_search():
    print("HERE", os.getcwd())
    with open("output/knr_errors.json", "r") as f:
        knr_errors = json.load(f)

    fig, axs = plt.subplots(1, len(knr_errors.keys()), figsize=(20, 5), sharey=True)
    fig.suptitle("K Nearest Neighbour Regression for different neighbourhood sizes")
    for i, (neighbours, lead_errors) in enumerate(knr_errors.items()):
        axs[i].plot(lead_times, lead_errors)
        axs[i].set_title(neighbours)

    plt.savefig("knr_lead_time.png")


if __name__ == "__main__":
    action = Action.PlotSearch

    key = "Temperature-StdPressureLev-1000.0"

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

    if action.is_search():
        search(config, key)

    if action.is_compare():
        compare(config)

    if action.is_plot_search():
        plot_search()

    exit(0)
    for group_A, group_D in zip(groups_A, groups_D):
        errors = {}
        train_times = {}
        predict_times = {}
        models = []

        model: Model
        for model in models:
            print(f"Running {model.name()}")
            model.run()
            error = model.mae("SurfAirTemp_A")
            errors[model.name()] = error
            train_times[model.name()] = model.train_time
            predict_times[model.name()] = model.predict_time

        df = pd.DataFrame.from_dict(
            {
                "method": list(errors.keys()),
                "error": [value for key, value in errors.items()],
                "train": [value for key, value in train_times.items()],
                "predict": [value for key, value in predict_times.items()],
            }
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(df, x="method", y="error")
        plt.title(f"Methods error comparison | {group_A.get_group()}")
        plt.ylabel(f"Mean Absolute Error (MAE) | {group_A.get_unit()}")
        plt.xlabel("Methods")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"figures/comparison_error_{group_A.value}_{group_D.value}_forecast.png"
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(df, x="method", y="train")
        plt.title(f"Methods training time comparison | {group_A.get_group()}")
        plt.ylabel(f"Training time (s)")
        plt.xlabel("Methods")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"figures/comparison_train_time_{group_A.value}_{group_D.value}_forecast.png"
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(df, x="method", y="predict")
        plt.title(f"Methods prediction time comparison | {group_A.get_group()}")
        plt.ylabel(f"Prediction time (s)")
        plt.xlabel("Methods")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"figures/comparison_prediction_time_{group_A.value}_{group_D.value}_forecast.png"
        )
