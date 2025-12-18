import argparse
import gc
import itertools
import json
import os
import warnings
from collections import defaultdict
from enum import Enum
from typing import Self

import cuml
import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
from cuml.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.functional import adaptive_avg_pool3d

from aggregate import get_time_data
from download import download_files
from gaussian_process import GaussianProcess
from k_nearest_neighbour import KNearestNeighbourRegression
from model import Config
from neural_network import MLP, WLSTM, NeuralNetwork

warnings.simplefilter(action="ignore", category=FutureWarning)


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
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, output),
        ),
        "N8",
    )

    n = NeuralNetwork(mlp, loss_function=nn.HuberLoss(), config=config, epoch=100)
    models.append(n)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output)
    gp = GaussianProcess(
        likelihood=likelihood,
        n_inducing_points=100,
        config=config,
        input=input,
        output=output,
    )
    models.append(gp)

    return models


def get_search_models(
    config: Config, input: int, output: int
) -> tuple[
    list[NeuralNetwork],
    list[KNearestNeighbourRegression],
    list[NeuralNetwork],
    list[GaussianProcess],
]:
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

    knr_models = []
    n_neighbours = [1, 2, 3, 4, 5, 10]
    knr = KNearestNeighbourRegression(
        knr=KNeighborsRegressor(n_neighbors=max(n_neighbours)),
        neighbours=n_neighbours,
        config=config,
    )
    knr_models.append(knr)

    mlp_models = []

    loss_functions = [nn.HuberLoss()]

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input),
            nn.LeakyReLU(),
            nn.Linear(input, input),
            nn.LeakyReLU(),
            nn.Linear(input, output),
        ),
        "N1",
    )
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.LeakyReLU(),
            nn.Linear(input * 2, input),
            nn.LeakyReLU(),
            nn.Linear(input, output),
        ),
        "N2",
    )
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.LeakyReLU(),
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
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Linear(input * 2, output),
        ),
        "N6",
    )
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, output),
        ),
        "N7",
    )
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, output),
        ),
        "N8",
    )
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, output),
        ),
        "N9",
    )
    # mlp_models.append(mlp)

    mlp = MLP(
        nn.Sequential(
            nn.Linear(input, input * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.BatchNorm1d(input * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, input * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input * 2, output),
        ),
        "N10",
    )
    # mlp_models.append(mlp)

    lstm = WLSTM(
        input // config.lag,
        1,
        input // 2,
        output,
        config.lag,
        input // config.lag,
        "LSTM-8-Linear",
    )
    mlp_models.append(lstm)

    nn_models = []
    for loss_function, network in itertools.product(loss_functions, mlp_models):
        network = NeuralNetwork(
            network=network, loss_function=loss_function, config=config, epoch=50
        )

        nn_models.append(network)

    inducing_points = [100]

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

    return lr_models, knr_models, nn_models, gp_models


class Action(Enum):
    Compare = "compare"
    Search = "search"
    Download = "download"
    PlotSearch = "plotsearch"
    PlotCompare = "plotcompare"
    PlotMap = "plotmap"
    ReadModels = "readmodels"

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

    def is_plot_search(self):
        return self.__matches__(Action.PlotSearch)

    def is_read_models(self):
        return self.__matches__(Action.ReadModels)


lead_max = 10


def plot_search(config):
    metrics = pd.read_csv("output/search_metrics.csv")

    for ignore in [
        # "N5-MSELoss",
        "GP-10",
        # "N5-HuberLoss",
        # "N6-L1Loss",
        # "N6-MSELoss",
    ]:
        metrics = metrics[(metrics["model"] != ignore)]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    sns.lineplot(
        data=metrics[metrics["key"] == "total"],
        x="lead",
        y="mae",
        hue="model",
        ax=axs[0],
    )
    axs[0].set_title("Total")
    axs[0].set_xlabel("Lead time")
    axs[0].set_ylabel("MAE")

    sns.lineplot(
        data=metrics[metrics["key"] == "total"],
        x="lead",
        y="rmse",
        hue="model",
        ax=axs[1],
    )
    axs[1].set_title("Total")
    axs[1].set_xlabel("Lead time")
    axs[1].set_ylabel("RMSE")

    sns.lineplot(
        data=metrics[metrics["key"] == "total"],
        x="lead",
        y="skill",
        hue="model",
        ax=axs[2],
    )
    axs[2].set_title("Total")
    axs[2].set_xlabel("Lead time")
    axs[2].set_ylabel("RMSE skill")

    for ax in axs:
        ax.legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    fig.suptitle(f"Neural Network Model Comparison")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    plt.savefig(f"figures/search_nn_total.png")


def track_hardware(
    df: pd.DataFrame,
    train_resources: dict,
    predict_resources: list[dict],
    name: str,
):
    df = df._append(
        {"model": name, "type": "train", **train_resources}, ignore_index=True
    )

    avg_predict_resource = {}
    for key in predict_resources[0].keys():
        avg_predict_resource[key] = 0
        for predict_resource in predict_resources:
            if "peak" in key:
                avg_predict_resource[key] = max(
                    avg_predict_resource[key], predict_resource[key]
                )
            else:
                avg_predict_resource[key] += predict_resource[key]

        if "peak" not in key:
            avg_predict_resource[key] /= len(predict_resources)

    df = df._append(
        {
            "model": name,
            "type": "predict",
            **avg_predict_resource,
        },
        ignore_index=True,
    )

    return df


def calculate_errors_per_key(
    model,
    scaler,
    output,
    y_test,
    ref,
    lead: int,
    keys: list[str],
    df: pd.DataFrame,
    name: str,
):
    print(f"Lead: {lead}")
    total_mae = 0.0
    total_sse_model = 0.0
    total_sse_ref = 0.0

    output_lead_rescaled = scaler.inverse_transform(output)
    y_test_rescaled = scaler.inverse_transform(y_test)
    ref_rescaled = scaler.inverse_transform(ref)
    for key in keys:
        mae = model.mae(key=key, y_test=y_test_rescaled, output=output_lead_rescaled)
        rmse = model.rmse(key=key, y_test=y_test_rescaled, output=output_lead_rescaled)
        skill = model.rmse_skill(
            key=key,
            ref=ref_rescaled,
            y_test=y_test_rescaled,
            output=output_lead_rescaled,
        )

        squared_error_model = model.squared_error(
            key=key, y_test=y_test_rescaled, output=output_lead_rescaled
        )
        squared_error_ref = model.squared_error(
            key=key, y_test=ref_rescaled, output=y_test_rescaled
        )
        y_test_rescaled = scaler.inverse_transform(y_test)

        total_mae += mae
        total_sse_model += squared_error_model
        total_sse_ref += squared_error_ref

        df = df._append(
            {
                "model": name,
                "lead": lead,
                "key": key,
                "mae": mae,
                "rmse": rmse,
                "skill": skill,
            },
            ignore_index=True,
        )

    return total_mae, total_sse_model, total_sse_ref, df


def calculate_total_errors(
    model, df, name, total_mae, total_sse_model, total_sse_ref, N, K
):
    for lead in range(1, lead_max + 1):
        mae = total_mae[lead - 1] / K
        sse_models = total_sse_model[lead - 1]
        sse_refs = total_sse_ref[lead - 1]

        rmse_model = np.sqrt(sse_models / (N * K))
        rmse_ref = np.sqrt(sse_refs / (N * K))

        skill = 1 - rmse_model / rmse_ref

        print(rmse_model, rmse_ref, skill)
        df = df._append(
            {
                "model": name,
                "lead": lead,
                "key": "total",
                "mae": mae,
                "rmse": rmse_model,
                "skill": skill,
            },
            ignore_index=True,
        )
    return df


class EvaluateType(Enum):
    Search = 0
    Compare = 1


def evaluate(
    type: EvaluateType,
    config: Config,
    ignore_lr=False,
    ignore_knr=False,
    ignore_nn=False,
    ignore_gp=False,
):
    error_metrics = pd.DataFrame(
        columns=["model", "lead", "key", "mae", "rmse", "skill"]
    )

    hardware_metrics = pd.DataFrame(
        columns=[
            "model",
            "type",
            "cpu_mean",
            "cpu_peak",
            "ram_mean",
            "ram_peak",
            "gpu_util_mean",
            "gpu_util_peak",
            "gpu_mem_mean",
            "gpu_mem_peak",
            "time",
        ]
    )

    if type == EvaluateType.Compare:
        train_file_keywords = [
            f"{config.train_year}.{month}" for month in config.months
        ]
        test_file_keywords = [f"{config.train_year}.{month}" for month in config.months]
    else:
        # train_file_keywords = f"{config.test_year}.{config.months[0]}"
        # test_file_keywords = f"{config.test_year}.{config.months[0]}"
        train_file_keywords = [
            f"{config.train_year}.{month}" for month in config.months
        ]
        test_file_keywords = [f"{config.train_year}.{month}" for month in config.months]

    config.leads = 1
    print("Getting train data")
    X_train, y_train, keys, shape = get_time_data(
        config, file_keyword=train_file_keywords
    )
    print(config.leads, X_train.shape, y_train.shape, len(keys))
    print(f"Keys: {keys} | Amount: {len(keys)}")
    quit()
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)

    config.leads = lead_max
    print("Getting test data")
    X_test, y_test, keys, shape = get_time_data(config, file_keyword=test_file_keywords)
    ref = X_test[:, -(len(keys) + 1) : -1]
    ref = y_scaler.transform(ref)
    print("Scaling test data")
    X_test = X_scaler.transform(X_test)
    for i in range(1, lead_max + 1):
        start = (i - 1) * len(keys)
        end = i * len(keys)
        y_test[:, start:end] = y_scaler.transform(y_test[:, start:end])

    hardware_metric_output_path = None
    if type == EvaluateType.Compare:
        error_metric_output_path = "output/compare_error_metrics.csv"
        hardware_metric_output_path = "output/compare_hardware_metrics.csv"

        lr, knr, network, gp = get_compare_models(config, X_train.shape[1], len(keys))
        lrs = [lr]
        knrs = [knr]
        networks = [network]
        gps = [gp]
    else:
        error_metric_output_path = "output/search_metrics.csv"
        lrs, knrs, networks, gps = get_search_models(
            config, X_train.shape[1], len(keys)
        )

    if os.path.exists(error_metric_output_path):
        error_metrics = pd.read_csv(error_metric_output_path)

    if hardware_metric_output_path is not None:
        if os.path.exists(hardware_metric_output_path):
            error_metrics = pd.read_csv(hardware_metric_output_path)
    #############################################################
    # Linear Regression
    #############################################################

    if not ignore_lr:
        for i, lr in enumerate(lrs):
            print(f"Running Linear Regression ({i}/{len(lrs)})")
            lr.keys = keys
            trainloader = lr.dataloader(X_train, y_train)
            lr.train(trainloader)

            y_ref = ref
            X_window = X_test
            y_window = y_test[:, : len(keys)]

            total_mae = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_model = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_ref = [0.0 for _ in range(1, lead_max + 1)]

            predict_resources = []

            for lead in range(1, lead_max + 1):
                testloader = lr.dataloader(X_window, y_window)
                output_lead = lr.predict(testloader)

                mae, sse_model, sse_ref, error_metrics = calculate_errors_per_key(
                    model=lr,
                    scaler=y_scaler,
                    ref=y_ref,
                    output=output_lead,
                    y_test=y_window,
                    lead=lead,
                    keys=keys,
                    df=error_metrics,
                    name="Linear Regression",
                )

                total_mae[lead - 1] = mae
                total_sse_model[lead - 1] = sse_model
                total_sse_ref[lead - 1] = sse_ref

                y_ref = y_window
                X_window = np.hstack((X_window[:, len(keys) :], output_lead))
                y_window = y_test[:, lead * len(keys) : (lead + 1) * len(keys)]

                predict_resources.append(lr.predict_resource)

            hardware_metrics = track_hardware(
                hardware_metrics,
                lr.train_resource,
                predict_resources,
                "Linear Regression",
            )

            n, _ = y_test.shape
            error_metrics = calculate_total_errors(
                model=lr,
                df=error_metrics,
                name="Linear Regression",
                total_mae=total_mae,
                total_sse_model=total_sse_model,
                total_sse_ref=total_sse_ref,
                N=n,
                K=len(keys),
            )

            if hardware_metric_output_path is not None:
                hardware_metrics.to_csv(hardware_metric_output_path, index=False)
            error_metrics.to_csv(error_metric_output_path, index=False)

        gc.collect()

    #############################################################
    # K Nearest Neighbours
    #############################################################

    if not ignore_knr:
        for i, knr in enumerate(knrs):
            print(f"Running K Nearest Neighbours ({i}/{len(knrs)})")
            knr.keys = keys

            knr.train(X_train, y_train)

            total_mae = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_model = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_ref = [0.0 for _ in range(1, lead_max + 1)]

            y_ref = ref
            X_window = X_test
            y_window = y_test[:, : len(keys)]

            predict_resources = []

            for lead in range(1, lead_max + 1):
                _, indices = knr.predict(X_window)
                output_lead = knr.predict_with_size(
                    knr.get_neighbour_sizes()[0], indices, y_train
                )

                mae, sse_model, sse_ref, error_metrics = calculate_errors_per_key(
                    model=knr,
                    scaler=y_scaler,
                    ref=y_ref,
                    output=output_lead,
                    y_test=y_window,
                    lead=lead,
                    keys=keys,
                    df=error_metrics,
                    name="K Nearest Neighbours",
                )

                total_mae[lead - 1] = mae
                total_sse_model[lead - 1] = sse_model
                total_sse_ref[lead - 1] = sse_ref

                y_ref = y_window
                X_window = np.hstack((X_window[:, len(keys) :], output_lead))
                y_window = y_test[:, lead * len(keys) : (lead + 1) * len(keys)]

                predict_resources.append(knr.predict_resource)

            hardware_metrics = track_hardware(
                hardware_metrics,
                knr.train_resource,
                predict_resources,
                "K Nearest Neighbours",
            )

            n, _ = y_test.shape
            error_metrics = calculate_total_errors(
                model=knr,
                df=error_metrics,
                name="K Nearest Neighbours",
                total_mae=total_mae,
                total_sse_model=total_sse_model,
                total_sse_ref=total_sse_ref,
                N=n,
                K=len(keys),
            )

            if hardware_metric_output_path is not None:
                hardware_metrics.to_csv(hardware_metric_output_path, index=False)
            error_metrics.to_csv(error_metric_output_path, index=False)

            gc.collect()

    #############################################################
    # Neural Network
    #############################################################

    if not ignore_nn:
        for i, network in enumerate(networks):
            if type == EvaluateType.Search:
                network_name = (
                    f"{network.get_network_name()}-{network.get_loss_function()}"
                )
            else:
                network_name = "Neural Network"

            network.keys = keys
            print(f"Running Neural Network ({network_name}, {i}/{len(networks)})")
            trainloader = network.dataloader(X_train, y_train)
            network.train(trainloader)

            y_ref = ref
            X_window = X_test
            y_window = y_test[:, : len(keys)]

            total_mae = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_model = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_ref = [0.0 for _ in range(1, lead_max + 1)]

            predict_resources = []

            for lead in range(1, lead_max + 1):
                testloader = network.dataloader(X_window, y_window)
                output_lead = network.predict(testloader)

                mae, sse_model, sse_ref, error_metrics = calculate_errors_per_key(
                    model=network,
                    scaler=y_scaler,
                    ref=y_ref,
                    output=output_lead,
                    y_test=y_window,
                    lead=lead,
                    keys=keys,
                    df=error_metrics,
                    name=network_name,
                )

                total_mae[lead - 1] = mae
                total_sse_model[lead - 1] = sse_model
                total_sse_ref[lead - 1] = sse_ref

                y_ref = y_window
                X_window = np.hstack((X_window[:, len(keys) :], output_lead))
                y_window = y_test[:, lead * len(keys) : (lead + 1) * len(keys)]

                predict_resources.append(network.predict_resource)

            hardware_metrics = track_hardware(
                hardware_metrics,
                network.train_resource,
                predict_resources,
                "Neural Network",
            )

            n, _ = y_test.shape
            error_metrics = calculate_total_errors(
                model=network,
                df=error_metrics,
                name=network_name,
                total_mae=total_mae,
                total_sse_model=total_sse_model,
                total_sse_ref=total_sse_ref,
                N=n,
                K=len(keys),
            )

            if hardware_metric_output_path is not None:
                hardware_metrics.to_csv(hardware_metric_output_path, index=False)
            error_metrics.to_csv(error_metric_output_path, index=False)

        gc.collect()

    #############################################################
    # Gaussian Process
    #############################################################

    if not ignore_gp:
        for i, gp in enumerate(gps):
            if type == EvaluateType.Search:
                gp_name = f"GP-{gp.get_inducing_points()}-100"
            else:
                gp_name = "Gaussian Process"

            gp.keys = keys
            print(f"Running Gaussian Process ({i}/{len(gps)})")
            print("Trainloading")
            trainloader, inducing_points = gp.dataloader(
                X_train, y_train, create_inducing_points=True
            )
            print("Creating model")
            gp.create_model(inducing_points)
            print("Training")
            gp.train(trainloader)

            y_ref = ref
            X_window = X_test
            y_window = y_test[:, : len(keys)]

            total_mae = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_model = [0.0 for _ in range(1, lead_max + 1)]
            total_sse_ref = [0.0 for _ in range(1, lead_max + 1)]

            predict_resources = []

            for lead in range(1, lead_max + 1):
                testloader, _ = gp.dataloader(X_window, y_window)
                output_lead = gp.predict(testloader)

                mae, sse_model, sse_ref, error_metrics = calculate_errors_per_key(
                    model=gp,
                    scaler=y_scaler,
                    ref=y_ref,
                    output=output_lead,
                    y_test=y_window,
                    lead=lead,
                    keys=keys,
                    df=error_metrics,
                    name=gp_name,
                )

                total_mae[lead - 1] = mae
                total_sse_model[lead - 1] = sse_model
                total_sse_ref[lead - 1] = sse_ref

                y_ref = y_window
                X_window = np.hstack((X_window[:, len(keys) :], output_lead))
                y_window = y_test[:, lead * len(keys) : (lead + 1) * len(keys)]

                predict_resources.append(gp.predict_resource)

            hardware_metrics = track_hardware(
                hardware_metrics,
                gp.train_resource,
                predict_resources,
                "Gaussian Process",
            )

            n, _ = y_test.shape
            error_metrics = calculate_total_errors(
                model=gp,
                df=error_metrics,
                name=gp_name,
                total_mae=total_mae,
                total_sse_model=total_sse_model,
                total_sse_ref=total_sse_ref,
                N=n,
                K=len(keys),
            )

            error_metrics.to_csv(error_metric_output_path, index=False)
            if hardware_metric_output_path is not None:
                hardware_metrics.to_csv(hardware_metric_output_path, index=False)


def read_models():
    lr_metrics = pd.read_csv("output/lr_metrics.csv")
    knr_metrics = pd.read_csv("output/knr_metrics.csv")
    nn_metrics = pd.read_csv("output/nn_metrics.csv")
    gp_metrics = pd.read_csv("output/gp_metrics.csv")

    mae = knr_metrics[knr_metrics["error_type"] == "mae"].sort_values(
        "error", ascending=True
    )
    rmse = knr_metrics[knr_metrics["error_type"] == "rmse"].sort_values(
        "error", ascending=True
    )
    skill = knr_metrics[knr_metrics["error_type"] == "skill"].sort_values(
        "error", ascending=False
    )

    print("K Nearest Neighbours Regression:\n", mae, "\n", rmse, "\n", skill)

    mae = nn_metrics[nn_metrics["error_type"] == "mae"].sort_values(
        "error", ascending=True
    )
    rmse = nn_metrics[nn_metrics["error_type"] == "rmse"].sort_values(
        "error", ascending=True
    )
    skill = nn_metrics[nn_metrics["error_type"] == "skill"].sort_values(
        "error", ascending=False
    )
    print("Neural Network:\n", mae, "\n", rmse, "\n", skill)

    mae = gp_metrics[gp_metrics["error_type"] == "mae"].sort_values(
        "error", ascending=True
    )
    rmse = gp_metrics[gp_metrics["error_type"] == "rmse"].sort_values(
        "error", ascending=True
    )
    skill = gp_metrics[gp_metrics["error_type"] == "skill"].sort_values(
        "error", ascending=False
    )
    print("Gaussian Process:\n", mae, "\n", rmse, "\n", skill)


def plot_compare():
    error_compare_metrics = pd.read_csv("output/compare_error_metrics.csv")
    hardware_compare_metrics = pd.read_csv("output/compare_hardware_metrics.csv")

    ###############################################################################
    # Temperature
    ###############################################################################
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

    # MAE

    g = sns.FacetGrid(
        error_compare_metrics[error_compare_metrics["key"].str.contains(main_key)],
        col="key",
        col_wrap=4,
        height=4,
        aspect=0.7,
    )
    g.map(sns.lineplot, "lead", "mae", "model")
    g.add_legend()
    g.set_axis_labels(x_var="Lead time (t)", y_var=f"MAE (Kelvin)")

    new_titles = {old: old.replace(f"{main_key}-", "") + " hPa" for old in g.col_names}

    for ax, old in zip(g.axes.flatten(), g.col_names):
        ax.set_title(new_titles[old])

    plt.savefig(f"figures/compare_models_mae_{main_key}.png")

    # RMSE

    g = sns.FacetGrid(
        error_compare_metrics[error_compare_metrics["key"].str.contains(main_key)],
        col="key",
        col_wrap=4,
        height=4,
        aspect=0.7,
    )
    g.map(sns.lineplot, "lead", "rmse", "model")
    g.add_legend()
    g.set_axis_labels(x_var="Lead time (t)", y_var="RMSE")

    new_titles = {old: old.replace(f"{main_key}-", "") + " hPa" for old in g.col_names}

    for ax, old in zip(g.axes.flatten(), g.col_names):
        ax.set_title(new_titles[old])

    plt.savefig(f"figures/compare_models_rmse_{main_key}.png")

    # RMSE skill

    g = sns.FacetGrid(
        error_compare_metrics[error_compare_metrics["key"].str.contains(main_key)],
        col="key",
        col_wrap=4,
        height=4,
        aspect=0.7,
    )
    g.map(sns.lineplot, "lead", "skill", "model")
    g.add_legend()
    g.set_axis_labels(x_var="Lead time (t)", y_var="RMSE skill")

    new_titles = {old: old.replace(f"{main_key}-", "") + " hPa" for old in g.col_names}

    for ax, old in zip(g.axes.flatten(), g.col_names):
        ax.set_title(new_titles[old])

    plt.savefig(f"figures/compare_models_skill_{main_key}.png")

    ###############################################################################
    # Humidity
    ###############################################################################

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

    # MAE

    g = sns.FacetGrid(
        error_compare_metrics[
            error_compare_metrics["key"].str.contains(f"{main_key}-")
        ],
        col="key",
        col_wrap=3,
        height=4,
        aspect=0.7,
    )
    g.map(sns.lineplot, "lead", "mae", "model")
    g.add_legend()
    g.set_axis_labels(x_var="Lead time (t)", y_var="MAE (%)")

    new_titles = {old: old.replace(f"{main_key}-", "") + " hPa" for old in g.col_names}

    for ax, old in zip(g.axes.flatten(), g.col_names):
        ax.set_title(new_titles[old])

    plt.savefig(f"figures/compare_models_mae_{main_key}.png")

    # RMSE

    g = sns.FacetGrid(
        error_compare_metrics[
            error_compare_metrics["key"].str.contains(f"{main_key}-")
        ],
        col="key",
        col_wrap=3,
        height=4,
        aspect=0.7,
    )
    g.map(sns.lineplot, "lead", "rmse", "model")
    g.add_legend()
    g.set_axis_labels(x_var="Lead time (t)", y_var="RMSE")

    new_titles = {old: old.replace(f"{main_key}-", "") + " hPa" for old in g.col_names}

    for ax, old in zip(g.axes.flatten(), g.col_names):
        ax.set_title(new_titles[old])

    plt.savefig(f"figures/compare_models_rmse_{main_key}.png")

    # RMSE skill

    g = sns.FacetGrid(
        error_compare_metrics[
            error_compare_metrics["key"].str.contains(f"{main_key}-")
        ],
        col="key",
        col_wrap=3,
        height=4,
        aspect=0.7,
    )
    g.map(sns.lineplot, "lead", "skill", "model")
    g.add_legend()
    g.set_axis_labels(x_var="Lead time (t)", y_var="RMSE skill")

    new_titles = {old: old.replace(f"{main_key}-", "") + " hPa" for old in g.col_names}

    for ax, old in zip(g.axes.flatten(), g.col_names):
        ax.set_title(new_titles[old])

    plt.savefig(f"figures/compare_models_skill_{main_key}.png")

    ###############################################################################
    # SurfAirTemp, RelHumSurf, SurfPres_Forecast.
    ###############################################################################

    for type in [("mae", "MAE"), ("rmse", "RMSE"), ("skill", "RMSE skill")]:
        key, name = type
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        sns.lineplot(
            data=error_compare_metrics[error_compare_metrics["key"] == "SurfAirTemp"],
            x="lead",
            y=key,
            hue="model",
            ax=axs[0],
        )
        axs[0].set_title("SurfAirTemp")
        axs[0].set_xlabel("Lead time")
        axs[0].set_ylabel(name)

        sns.lineplot(
            data=error_compare_metrics[error_compare_metrics["key"] == "RelHumSurf"],
            x="lead",
            y=key,
            hue="model",
            ax=axs[1],
        )
        axs[1].set_title("RelHumSurf")
        axs[1].set_xlabel("Lead time")
        axs[1].set_ylabel(name)

        sns.lineplot(
            data=error_compare_metrics[
                error_compare_metrics["key"] == "SurfPres_Forecast"
            ],
            x="lead",
            y=key,
            hue="model",
            ax=axs[2],
        )
        axs[2].set_title("SurfPres_Forecast")
        axs[2].set_xlabel("Lead time")
        axs[2].set_ylabel(name)

        for ax in axs:
            ax.legend().remove()

        handles, labels = axs[0].get_legend_handles_labels()
        fig.suptitle(f"Model Comparison")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        print(handles, labels)
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))
        plt.savefig(f"figures/compare_models_{key}_main_3.png")

    ###############################################################################
    # Total
    ###############################################################################

    print(error_compare_metrics[error_compare_metrics["key"] == "total"])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.lineplot(
        data=error_compare_metrics[error_compare_metrics["key"] == "total"],
        x="lead",
        y="mae",
        hue="model",
        ax=axs[0],
    )
    axs[0].set_title("Total MAE")
    axs[0].set_xlabel("Lead time")
    axs[0].set_ylabel("MAE")

    sns.lineplot(
        data=error_compare_metrics[error_compare_metrics["key"] == "total"],
        x="lead",
        y="rmse",
        hue="model",
        ax=axs[1],
    )
    axs[1].set_title("Total RMSE")
    axs[1].set_xlabel("Lead time")
    axs[1].set_ylabel("RMSE")

    """
    sns.lineplot(
        data=compare_metrics[compare_metrics["key"] == "total"],
        x="lead",
        y="skill",
        hue="model",
        ax=axs[2],
    )
    axs[2].set_title("Total")
    axs[2].set_xlabel("Lead time")
    axs[2].set_ylabel("RMSE skill")
    """

    for ax in axs:
        ax.legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    fig.suptitle(f"Model Comparison")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    plt.savefig(f"figures/compare_models_total.png")

    ###############################################################################
    # Hardware
    ###############################################################################
    train_hardware = hardware_compare_metrics[
        hardware_compare_metrics["type"] == "train"
    ]
    predict_hardware = hardware_compare_metrics[
        hardware_compare_metrics["type"] == "predict"
    ]

    name_mappings = {
        "train_cpu_mean": "Train Mean",
        "train_cpu_peak": "Train Peak",
        "predict_cpu_mean": "Predict Mean",
        "predict_cpu_peak": "Predict Peak",
        "train_gpu_util_mean": "Train Mean",
        "train_gpu_util_peak": "Train Peak",
        "predict_gpu_util_mean": "Predict Mean",
        "predict_gpu_util_peak": "Predict Peak",
        "train_gpu_mem_mean": "Train Mean",
        "train_gpu_mem_peak": "Train Peak",
        "predict_gpu_mem_mean": "Predict Mean",
        "predict_gpu_mem_peak": "Predict Peak",
        "train_ram_mean": "Train Mean",
        "train_ram_peak": "Train Peak",
        "predict_ram_mean": "Predict Mean",
        "predict_ram_peak": "Predict Peak",
    }

    for y, title, label in [
        ("cpu", "CPU usage", "Usage (%)"),
        ("ram", "RAM usage", "MB"),
        ("gpu_util", "GPU Utility usage", "Usage (%)"),
        ("gpu_mem", "GPU Memory usage", "MB"),
    ]:
        train_hardware_melt = hardware_compare_metrics.melt(
            id_vars=["model", "type"],
            value_vars=[f"{y}_mean", f"{y}_peak"],
            var_name="stat",
            value_name="value",
        )

        print(train_hardware_melt)
        y_train_hardware_melt = train_hardware_melt[
            (train_hardware_melt["stat"] == f"{y}_mean")
            | (train_hardware_melt["stat"] == f"{y}_peak")
        ]
        print(y_train_hardware_melt)
        y_train_hardware_melt["stat"] = (
            y_train_hardware_melt["type"] + "_" + y_train_hardware_melt["stat"]
        )
        y_train_hardware_melt["stat"] = y_train_hardware_melt["stat"].replace(
            name_mappings
        )
        print(y_train_hardware_melt)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=y_train_hardware_melt,
            x="model",
            y="value",
            hue="stat",
            hue_order=["Train Mean", "Train Peak", "Predict Mean", "Predict Peak"],
            ax=ax,
        )
        ax.set_title(f"Model {title} comparison")
        ax.set_ylabel(label)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(f"figures/compare_models_{y}.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=train_hardware, x="model", y="time", ax=ax)
    ax.set_title("Model train time comparison")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("figures/compare_models_train_time.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=predict_hardware, x="model", y="time", ax=ax)
    ax.set_title("Model predict time comparison")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("figures/compare_models_predict_time.png")


def plot_map(config):
    file = os.listdir("./data")[0]
    dataset = xr.open_dataset(f"./data/{file}")
    lon_low, lon_high = config.lon_low + 180, config.lon_high + 180
    lat_low, lat_high = (
        90 - config.lat_high,
        90 - config.lat_low,
    )

    area = dataset["LandSeaMask"][lat_low:lat_high, lon_low:lon_high]
    print(area)
    # area = np.flipud(area.to_numpy())
    print(area.shape)
    print(area[0, 0])
    map = matplotlib.colors.ListedColormap(["#FFFFFF", "#02367B"])
    plt.imshow(area, cmap=map, origin="upper", extent=[-10, 50, 30, 80], vmin=0, vmax=1)
    plt.grid(None)
    plt.xlabel("° Latitude")
    plt.ylabel("° Longitude")
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
        lon_low = -10
        lon_high = 50
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
            evaluate(
                EvaluateType.Search,
                config,
                ignore_lr=ignore_lr,
                ignore_nn=ignore_nn,
                ignore_knr=ignore_knr,
                ignore_gp=ignore_gp,
            )

        if action.is_compare():
            print("Compare")
            evaluate(
                EvaluateType.Compare,
                config,
                ignore_lr=ignore_lr,
                ignore_nn=ignore_nn,
                ignore_knr=ignore_knr,
                ignore_gp=ignore_gp,
            )

        if action.is_plot_compare():
            print("Plot Compare")
            plot_compare()

        if action.is_plot_search():
            print("Plot Search")
            plot_search(config)

        if action.is_read_models():
            print("Read models")
            read_models()
    else:
        print("Please provide --type")
        exit(1)
