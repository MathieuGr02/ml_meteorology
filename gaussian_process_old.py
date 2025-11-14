import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import gpytorch
import torch
from torch import nn

import itertools

from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(gp, likelihood, trainloader, lr=1e-2, epoch=20) -> None:
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)

    gp.cuda().train()
    likelihood.cuda().train()

    loss_function = gpytorch.mlls.VariationalELBO(
        likelihood, gp, num_data=len(trainloader.dataset)
    ).cuda()

    best_loss = float("inf")
    no_improvement = 0

    for epoch_i in range(epoch):
        total_loss = 0
        minibatch_iter = tqdm(trainloader, desc="Minibatch", leave=False)
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=True, log_prob=True, solves=False
        ):
            for X, y in minibatch_iter:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                output = gp(X)

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


def predict(gp, likelihood, testloader) -> float:
    gp.eval()
    likelihood.eval()

    losses = []
    loss_function = nn.L1Loss()

    with torch.no_grad() and gpytorch.settings.fast_pred_var():
        for X, y in testloader:
            X, y = X.to(device).float(), y.to(device).float()

            outputs = likelihood(gp(X))
            l = loss_function(outputs.mean, y)
            losses.append(l.item())

    avg = np.average(losses)
    return avg


def prepare_data(group, amount_inducing_points: int = 1000):
    from aggregate import get_data

    training, training_target, test, test_target = get_data(group)

    inducing_points = torch.tensor(
        training[
            np.random.choice(training.shape[0], amount_inducing_points, replace=False),
            :,
        ],
        dtype=torch.float32,
    ).to(device)

    training = torch.tensor(training, dtype=torch.float32).to(device)
    training_target = torch.tensor(training_target, dtype=torch.float32).to(device)
    test = torch.tensor(test, dtype=torch.float32).to(device)
    test_target = torch.tensor(test_target, dtype=torch.float32).to(device)

    train_data = MeteoData(training, training_target)
    test_data = MeteoData(test, test_target)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=1024, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=1024, shuffle=True, num_workers=0
    )
    return inducing_points, trainloader, testloader


class MeteoData:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    print("-- Gaussian Process --")
    from aggregate import DataGroups

    groups = [
        DataGroups.SurfaceAirTemperatureA,
        DataGroups.OzoneA,
        DataGroups.RelativeHumiditySurfaceA,
    ]

    for group in groups:
        read = True

        if read:
            print(group.value)
            data = pd.read_csv(f"gp_results_gridsearch_{group.value}.csv")
            # print(data)
            argmin = data["loss"].argmin()
            print(data.iloc[argmin])

            loss_group = data.loc[
                data.groupby(["amount inducing points"])["loss"].idxmin()
            ].reset_index(drop=True)

            loss_group = loss_group[["loss", "amount inducing points"]]

            plt.figure(figsize=(8, 5))
            sns.lineplot(loss_group, y="loss", x="amount inducing points")
            plt.title("Optimal loss w.r.t. amount of inducing points")
            plt.xlabel("Inducing points")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(f"figures/gp_opt_grid_search_{group.value}.png")
            # plt.show()
        else:
            results = []

            lrs = [1e-3, 1e-2, 1e-1]
            epochs = [20, 30]
            inducing_points = [250, 500, 1000]

            print("Getting data")

            length = len(lrs) * len(epochs) * len(inducing_points)
            for i, (lr, n_epochs, n_inducing_points) in enumerate(
                tqdm(
                    itertools.product(lrs, epochs, inducing_points),
                    total=length,
                    desc="Grid Search",
                )
            ):
                print(f"{i}/{length}")
                inducing_points, trainloader, testloader = prepare_data(
                    group, n_inducing_points
                )

                gp = GP(inducing_points)
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                train(gp, likelihood, trainloader, lr=lr, epoch=n_epochs)
                error = predict(gp, likelihood, testloader)

                results.append(
                    {
                        "lr": lr,
                        "epochs": n_epochs,
                        "amount inducing points": n_inducing_points,
                        "loss": error,
                    }
                )

                pd.DataFrame(results).to_csv(
                    f"gp_results_gridsearch_{group.value}.csv", index=False
                )

            # plt.figure(figsize=(10, 5))
            # sns.barplot({"gp": error})
            # plt.title("Linear Regression Polynomial degree")
            # plt.ylabel("MAE")
            # plt.xlabel("Polynomial degree")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f"figures/gp_opt_{group}.png")
