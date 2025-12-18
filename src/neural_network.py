from typing import Any, override

import numpy as np
import seaborn as sns
import torch
from torch import nn
from tqdm import tqdm

from model import Config, Model
from resource_tracker import ResourceTracker

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
        lr: float = 0.01,
        epoch: int = 100,
        batch_size: int = 2048,
        optimizer=None,
    ):
        super().__init__(config)
        self.network = network
        self.lr = lr
        self.epoch = epoch
        self.loss_function = loss_function
        self.batch_size = batch_size
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=self.lr, weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer

    def name(self) -> str:
        return "Neural Network"

    def get_network_name(self) -> str:
        return self.network.__str__()

    def get_learning_rate(self) -> float:
        return self.lr

    def get_epoch(self) -> int:
        return self.epoch

    def get_loss_function(self):
        return type(self.loss_function).__name__

    def run(self, X_train, y_train, X_test, y_test):
        trainloader = self.dataloader(X_train, y_train)
        testloader = self.dataloader(X_test, y_test)

        self.train(trainloader)
        self.predict(testloader)

    @override
    def train(self, X, y=None):
        with ResourceTracker() as rt:
            self._train(X)

        self.train_resource = rt.results()

    @override
    def predict(self, X):
        with ResourceTracker() as rt:
            outputs = self._predict(X)

        self.predict_resource = rt.results()
        return outputs

    def _train(self, trainloader):
        self.network.to(device)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.90)

        for epoch in range(self.epoch):
            epoch_loss = 0.0
            n_batches = 0
            minibatch_iter = tqdm(trainloader, desc="Minibatch", leave=False)
            for X, y in minibatch_iter:
                X, y = X.to(device).float(), y.to(device).float()
                self.optimizer.zero_grad()

                outputs = self.network(X)

                loss = self.loss_function(outputs.squeeze(), y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            print(
                f"(lr: {scheduler._last_lr[-1]}) Epoch {epoch}: loss: {epoch_loss / n_batches:.4f}"
            )
            scheduler.step()

    def _predict(self, testloader):
        self.network.eval()

        outputs = ()

        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device).float(), y.to(device).float()
                outputs += (self.network(X).cpu(),)

        return np.row_stack(outputs)

    def dataloader(self, X, y) -> Any:
        data = MeteoData(X, y)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, num_workers=10
        )
        return loader


class WLSTM(nn.Module):
    def __init__(self, features, layers, hidden, output, n, m, name) -> None:
        super().__init__()
        self.n = n
        self.m = m
        self.name = name

        self.linear_in = nn.Linear(n * m, n * m)
        self.ReLU = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2,
        )

        self.linear_out = nn.Linear(hidden, output)

    def __str__(self):
        return self.name

    def forward(self, x):
        x = self.linear_in(x)
        x = self.ReLU(x)

        B = x.size(0)
        x = x.view(B, self.n, self.m)

        out, _ = self.lstm(x)

        last_hidden = out[:, -1, :]
        return self.linear_out(last_hidden)


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
