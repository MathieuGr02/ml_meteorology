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
    def __init__(self, points, num_inputs: int, num_tasks: int):
        num_latents = num_tasks // 2

        print(points.shape)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            points.size(0), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
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
    input: int
    output: int

    def __init__(
        self,
        likelihood,
        n_inducing_points,
        config: Config,
        input: int,
        output: int,
        lr: float = 0.1,
        epoch: int = 50,
    ):
        super().__init__(config)
        self.lr = lr
        self.epoch = epoch
        self.n_inducing_points = n_inducing_points
        self.likelihood = likelihood
        self.input = input
        self.output = output

    def name(self) -> str:
        return "Gaussian Process"

    def get_inducing_points(self) -> int:
        return self.n_inducing_points

    def get_learning_rate(self) -> float:
        return self.lr

    def get_epoch(self) -> int:
        return self.epoch

    def trainloader(self):
        X_train, y_train, self.keys, self.shape = super().get_train_data()

        self.inducing_points = torch.tensor(
            X_train[
                np.random.choice(
                    X_train.shape[0], self.n_inducing_points, replace=False
                ),
                :,
            ],
            dtype=torch.float32,
        ).to(device)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        train_data = MeteoData(X_train, y_train)

        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=256, shuffle=True, num_workers=0
        )

        return self.inducing_points, trainloader

    def testloader(self):
        self.X_test, self.y_test, *_ = super().get_test_data()

        X_test = torch.tensor(self.X_test, dtype=torch.float32)
        y_test = torch.tensor(self.y_test, dtype=torch.float32)

        test_data = MeteoData(X_test, y_test)

        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=256, shuffle=True, num_workers=0
        )

        return testloader

    def create_model(self):
        self.gp = GP(self.inducing_points, num_inputs=self.input, num_tasks=self.output)
        self.gp.cuda().train()
        self.likelihood.cuda().train()

    def run(self):
        self.inducing_points, trainloader = self.trainloader()
        testloader = self.testloader()

        self.create_model()

        self.train(trainloader)
        self.predict(testloader)

    @override
    def train(self, X, y=None):
        self.train_time, _ = track_time(lambda: self._train(X))

    @override
    def predict(self, X):
        self.predict_time, _ = track_time(lambda: self._predict(X))

    def _train(self, trainloader):
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        loss_function = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=len(trainloader.dataset)
        ).cuda()

        for epoch_i in range(self.epoch):
            minibatch_iter = tqdm(trainloader, desc="Minibatch", leave=False)

            with (
                gpytorch.settings.fast_computations(
                    covar_root_decomposition=True, log_prob=True, solves=False
                ),
                gpytorch.settings.max_root_decomposition_size(50),
                gpytorch.settings.max_preconditioner_size(0),
            ):
                for X, y in minibatch_iter:
                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()

                    output = self.gp(X)

                    loss = -loss_function(output, y)

                    loss.backward()
                    optimizer.step()

            scheduler.step()

    def _predict(self, testloader):
        self.gp.eval()
        self.likelihood.eval()

        outputs = ()

        with torch.no_grad() and gpytorch.settings.fast_pred_var():
            for X, y in testloader:
                X, y = X.to(device).float(), y.to(device).float()
                outputs += (self.likelihood(self.gp(X)).mean.cpu().detach(),)

        self.outputs = np.row_stack(outputs)
