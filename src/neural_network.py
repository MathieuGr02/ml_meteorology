import itertools
from collections.abc import Callable
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.optim.optimizer import Kwargs
from tqdm import tqdm

from model import Config, Model
from utils import track_time

sns.set_theme()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(Model):
    lr: float
    epoch: int
    batch_size: int

    def __init__(
        self,
        network,
        loss_function,
        config: Config,
        lr: float = 0.1,
        epoch: int = 50,
        batch_size: int = 512,
        optimizer=None,
    ):
        super().__init__(config)
        self.network = network
        self.lr = lr
        self.epoch = epoch
        self.loss_function = loss_function
        self.batch_size = batch_size
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer

    def name(self) -> str:
        return "Neural Network"

    def network_name(self) -> str:
        return self.network.__str__()

    def get_learning_rate(self) -> float:
        return self.lr

    def get_epoch(self) -> int:
        return self.epoch

    def get_loss_function(self):
        return type(self.loss_function).__name__

    def run(self):
        trainloader = self.trainloader()
        testloader = self.trainloader()

        self.train(trainloader)
        self.predict(testloader)

    @override
    def train(self, X, y=None):
        self.train_time, _ = track_time(lambda: self._train(X))

    @override
    def predict(self, X):
        self.predict_time, _ = track_time(lambda: self._predict(X))

    def _train(self, trainloader):
        self.network.to(device)

        epoch_losses = []

        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        for epoch in range(self.epoch):
            running_loss = 0
            count = 0
            minibatch_iter = tqdm(trainloader, desc="Minibatch", leave=False)
            for X, y in minibatch_iter:
                X, y = X.to(device).float(), y.to(device).float()
                self.optimizer.zero_grad()

                outputs = self.network(X)

                loss = self.loss_function(outputs.squeeze(), y)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                count += 1

            scheduler.step()

            avg_loss = running_loss / count
            epoch_losses.append(avg_loss)

    def _predict(self, testloader):
        self.network.eval()

        outputs = ()

        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device).float(), y.to(device).float()
                outputs += (self.network(X).cpu(),)

        self.outputs = np.row_stack(outputs)

    def trainloader(self) -> Any:
        X_train, y_train, *_ = super().get_train_data()
        train_data = MeteoData(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=10
        )
        return trainloader

    def testloader(self) -> Any:
        self.X_test, self.y_test, *_ = super().get_test_data()
        test_data = MeteoData(self.X_test, self.y_test)
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=self.batch_size, shuffle=True, num_workers=10
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
