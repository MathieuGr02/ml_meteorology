from typing import override

import gpytorch
import numpy as np
import torch
from tqdm import tqdm

from model import Config, MeteoData, Model
from resource_tracker import ResourceTracker

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
        batch_size: int = 1024,
        lr: float = 0.1,
        epoch: int = 30,
    ):
        super().__init__(config)
        self.lr = lr
        self.epoch = epoch
        self.n_inducing_points = n_inducing_points
        self.likelihood = likelihood
        self.input = input
        self.output = output
        self.batch_size = batch_size

    def name(self) -> str:
        return "Gaussian Process"

    def get_inducing_points(self) -> int:
        return self.n_inducing_points

    def get_learning_rate(self) -> float:
        return self.lr

    def get_epoch(self) -> int:
        return self.epoch

    def dataloader(self, X, y, create_inducing_points: bool = False):
        inducing_points = None
        if create_inducing_points:
            inducing_points = torch.tensor(
                X[
                    np.random.choice(X.shape[0], self.n_inducing_points, replace=False),
                    :,
                ],
                dtype=torch.float32,
            ).to(device)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        data = MeteoData(X, y)

        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, num_workers=10
        )

        return loader, inducing_points

    def create_model(self, inducing_points):
        self.gp = GP(inducing_points, num_inputs=self.input, num_tasks=self.output)
        self.gp.cuda().train()
        self.likelihood.cuda().train()

    def run(self, X_train, y_train, X_test, y_test):
        trainloader, inducing_points = self.dataloader(
            X_train, y_train, create_inducing_points=True
        )
        testloader = self.dataloader(X_test, y_test)

        self.create_model()

        self.train(trainloader)
        self.predict(testloader)

    @override
    def train(self, data):
        with ResourceTracker() as rt:
            self._train(data)

        self.train_resource = rt.results()

    @override
    def predict(self, X):
        with ResourceTracker() as rt:
            outputs = self._predict(X)

        self.predict_resource = rt.results()
        return outputs

    def _train(self, trainloader):
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        loss_function = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=len(trainloader.dataset)
        ).cuda()

        for epoch in range(self.epoch):
            epoch_loss = 0.0
            n_batches = 0

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

                    epoch_loss += loss.item()
                    n_batches += 1

                print(
                    f"(lr: {scheduler._last_lr[-1]}) Epoch {epoch}: loss: {epoch_loss / n_batches:.4f}"
                )

            scheduler.step()

    def _predict(self, testloader):
        self.gp.eval()
        self.likelihood.eval()

        outputs = ()

        with torch.no_grad() and gpytorch.settings.fast_pred_var():
            for X, y in testloader:
                X, y = X.to(device).float(), y.to(device).float()
                outputs += (self.likelihood(self.gp(X)).mean.cpu().detach(),)

        return np.row_stack(outputs)
