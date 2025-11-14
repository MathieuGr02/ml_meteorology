import itertools
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from tqdm import tqdm

from model import Config, Model
from utils import track_time

sns.set_theme()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(Model):
    def __init__(self, network, loss_function, lr: float, epoch: int, config: Config):
        super().__init__(config)
        self.network = network
        self.lr = lr
        self.epoch = epoch
        self.loss_function = loss_function

    def name(self) -> str:
        return "Neural Network"

    def network_name(self) -> str:
        return self.network.__str__()

    def run(self):
        trainloader = self.train_data()
        testloader = self.test_data()

        self.train_time, _ = track_time(lambda: self._train(trainloader))
        self.predict_time, _ = track_time(lambda: self._predict(testloader))

    def _train(self, trainloader):
        self.network.to(device)

        epoch_losses = []

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        for epoch in range(self.epoch):
            running_loss = 0
            count = 0

            for i, data in enumerate(trainloader, 0):
                X, y = data
                X, y = X.to(device).float(), y.to(device).float()
                optimizer.zero_grad()

                outputs = self.network(X)

                loss = self.loss_function(outputs.squeeze(), y)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                count += 1

            avg_loss = running_loss / count
            epoch_losses.append(avg_loss)

    def _predict(self, testloader):
        self.network.eval()

        outputs = ()

        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device).float(), y.to(device).float()
                outputs += (self.network(X).cpu(),)

        self.output = np.row_stack(outputs)

    @override
    def train_data(self) -> Any:
        X_train, y_train, self.keys, _ = super().train_data()
        train_data = MeteoData(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=512, shuffle=True, num_workers=10
        )
        return trainloader

    @override
    def test_data(self) -> Any:
        self.X_test, self.y_test, *_ = super().test_data()
        test_data = MeteoData(self.X_test, self.y_test)
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=512, shuffle=True, num_workers=10
        )
        return testloader


class MLP(nn.Module):
    def __init__(self, seq, name):
        super().__init__()
        self.name = name
        self.layers = seq

    def __str__(self):
        return self.name

    def forward(self, x):
        return self.layers(x)


class MeteoData:
    def __init__(self, X, y) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_features(self) -> int:
        return self.X.shape[1]


def prepare_data(group):
    from aggregate import get_data

    print("Getting data")
    training, training_target, test, test_target = get_data(group)

    print(f"Shape {training.shape}")
    test_data = MeteoData(test, test_target)

    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=512, shuffle=True, num_workers=10
    )

    return trainloader, testloader


def get_network(name: str, features: int):
    match name:
        case "N1":
            return MLP(
                nn.Sequential(
                    nn.Linear(features, features),
                    nn.ReLU(),
                    nn.Linear(features, features),
                    nn.ReLU(),
                    nn.Linear(features, 1),
                ),
                "N1",
            )
        case "N2":
            return MLP(
                nn.Sequential(
                    nn.Linear(features, features * 2),
                    nn.ReLU(),
                    nn.Linear(features * 2, features),
                    nn.ReLU(),
                    nn.Linear(features, 1),
                ),
                "N2",
            )
        case "N3":
            return MLP(
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
        case "N4":
            return MLP(
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


def reset_weights(network):
    for layer in network.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


if __name__ == "__main__":
    print("-- Neural Network --")
    from aggregate import DataGroups

    # group = DataGroups.SurfaceAirTemperatureA

    groups = [
        DataGroups.SurfaceAirTemperatureA,
        DataGroups.OzoneA,
        DataGroups.RelativeHumiditySurfaceA,
    ]
    for group in groups:
        read = True

        if read:
            print(group.value)
            data = pd.read_csv(f"nn_results_gridsearch_{group.value}.csv")
            argmin = data["val_loss"].argmin()
            print(data.iloc[argmin])

            loss_group = data.loc[
                data.groupby(["network", "loss_fn"])["val_loss"].idxmin()
            ].reset_index(drop=True)

            pivot = loss_group.pivot(
                index="network", columns="loss_fn", values="val_loss"
            )

            plt.figure(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, cmap="plasma", fmt=".4f")
            plt.title("Optimal loss for model w.r.t. loss function")
            plt.xlabel("Loss Function")
            plt.ylabel("Network")
            plt.tight_layout()
            plt.savefig(f"figures/nn_opt_grid_search_{group.value}.png")
            # plt.show()
        else:
            trainloader, testloader = prepare_data(group)
            print(trainloader.dataset)

            features = trainloader.dataset.get_features()
            print(f"Amount of features: {features}")

            lrs = [1e-3, 1e-2, 1e-1]
            epochs = [20, 30, 40]
            loss_fns = [nn.L1Loss(), nn.MSELoss(), nn.HuberLoss()]
            loss_names = ["L1", "MSE", "Huber"]

            network_names = ["N1", "N2", "N3", "N4"]

            results = []
            for network_name, (lr, n_epochs, (loss_fn, loss_name)) in tqdm(
                itertools.product(
                    network_names,
                    itertools.product(lrs, epochs, zip(loss_fns, loss_names)),
                ),
                total=len(network_names) * len(lrs) * len(epochs) * len(loss_fns),
            ):
                network = get_network(network_name, features)

                print(f"{group.value} {network_name} {n_epochs} {lr} {loss_name} ")
                network.to(device)
                train(network, trainloader, loss_fn, lr=lr, epoch=n_epochs)
                error = predict(network, testloader)

                results.append(
                    {
                        "network": network_name,
                        "lr": lr,
                        "epochs": n_epochs,
                        "loss_fn": loss_name,
                        "val_loss": error,
                    }
                )

                pd.DataFrame(results).to_csv(
                    f"nn_results_gridsearch_{group.value}.csv", index=False
                )

            # plt.figure(figsize=(10, 5))
            # sns.barplot(results_networks)
            # plt.title(f"Optimal Neural Networks comparison | {group.get_group()}")
            # plt.ylabel(f"Mean Absolute Error (MAE) | {group.get_unit()}")
            # plt.xlabel("Network")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f"figures/nn_opt_{group}_6_networks.png")
            # plt.show()
