from typing import override

import gpytorch
import numpy as np
import torch
from cuml.metrics import mean_absolute_error
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from torch import nn
from tqdm import tqdm

from model import Config, MeteoData, Model
from utils import track_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GP(gpytorch.models.ApproximateGP):
    def __init__(self, points, num_tasks):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            points.size(0)
        )

        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


"""
class GP(gpytorch.models.ApproximateGP):
    def __init__(self, points, features):
        print(points.shape)
        variational_distribution = CholeskyVariationalDistribution(
            points.size(0), num_features=features
        )
        base_strategy = VariationalStrategy(
            self,
            points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        variational_strategy = IndependentMultitaskVariationalStrategy(
            base_strategy, num_tasks=features
        )

        super(GP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=features
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=features, rank=1
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)

"""


class GaussianProcess(Model):
    epoch: int
    lr: float

    def __init__(
        self,
        likelihood,
        lr: float,
        epoch: int,
        n_inducing_points,
        config: Config,
    ):
        super().__init__(config)
        self.lr = lr
        self.epoch = epoch
        self.n_inducing_points = n_inducing_points
        self.likelihood = likelihood

    def name(self) -> str:
        return "Gaussian Process"

    def inducing_points(self) -> int:
        return self.n_inducing_points

    def get_learning_rate(self) -> float:
        return self.lr

    def get_epoch(self) -> int:
        return self.epoch

    @override
    def train_data(self):
        X_train, y_train, self.keys, self.shape = super().train_data()

        inducing_points = torch.tensor(
            X_train[
                np.random.choice(
                    X_train.shape[0], self.n_inducing_points, replace=False
                ),
                :,
            ],
            dtype=torch.float32,
        ).to(device)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

        train_data = MeteoData(X_train, y_train)

        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=1024, shuffle=True, num_workers=0
        )

        return inducing_points, trainloader

    @override
    def test_data(self):
        self.X_test, self.y_test, *_ = super().test_data()

        X_test = torch.tensor(self.X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(self.y_test, dtype=torch.float32).to(device)

        test_data = MeteoData(X_test, y_test)

        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=1024, shuffle=True, num_workers=0
        )

        return testloader

    def run(self):
        inducing_points, trainloader = self.train_data()
        testloader = self.test_data()

        self.gp = GP(inducing_points, len(self.keys))
        self.gp.cuda().train()
        self.likelihood.cuda().train()

        self.train_time, _ = track_time(lambda: self._train(trainloader))
        self.predict_time, _ = track_time(lambda: self._predict(testloader))

    def _train(self, trainloader):
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)

        loss_function = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=len(trainloader.dataset)
        ).cuda()

        best_loss = float("inf")
        no_improvement = 0

        for epoch_i in range(self.epoch):
            total_loss = 0
            minibatch_iter = tqdm(trainloader, desc="Minibatch", leave=False)
            with gpytorch.settings.fast_computations(
                covar_root_decomposition=True, log_prob=True, solves=False
            ):
                for X, y in minibatch_iter:
                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()
                    output = self.gp(X)
                    loss = -loss_function(output, y)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
            avg_loss = total_loss / len(trainloader)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= 5:
                return

    def _predict(self, testloader):
        self.gp.eval()
        self.likelihood.eval()

        outputs = ()

        with torch.no_grad() and gpytorch.settings.fast_pred_var():
            for X, y in testloader:
                X, y = X.to(device).float(), y.to(device).float()
                outputs += (self.likelihood(self.gp(X)).mean.cpu().detach(),)

        self.output = np.row_stack(outputs)
