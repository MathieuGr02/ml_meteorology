from enum import Enum
from cuml.neighbors import KNeighborsRegressor
from cuml.preprocessing import PolynomialFeatures
from cuml.linear_model import LinearRegression as cuMLLinearRegression
from sklearn.linear_model import LinearRegression
from torch import nn
import neural_network
import gaussian_process


class DataGroups(Enum):
    SurfaceAirTemperatureA = "surface_air_temperature_A"
    SurfaceAirTemperatureD = "surface_air_temperature_D"
    TimeSurfaceAirTemperatureA = "time_surface_air_temperature_A"
    TimeSurfaceAirTemperatureD = "time_surface_air_temperature_D"
    OzoneA = "ozone_A"
    OzoneD = "ozone_D"
    TimeOzoneA = "time_ozone_A"
    TimeOzoneD = "time_ozone_D"
    RelativeHumiditySurfaceA = "relative_humidity_surface_A"
    RelativeHumiditySurfaceD = "relative_humidity_surface_D"
    TimeRelativeHumiditySurfaceA = "time_relative_humidity_surface_A"
    TimeRelativeHumiditySurfaceD = "time_relative_humidity_surface_D"

    def get_group(self) -> str:
        match self:
            case DataGroups.SurfaceAirTemperatureA | DataGroups.SurfaceAirTemperatureD:
                return "Group 1"
            case DataGroups.OzoneA | DataGroups.OzoneD:
                return "Group 2"
            case (
                DataGroups.RelativeHumiditySurfaceA
                | DataGroups.RelativeHumiditySurfaceD
            ):
                return "Group 3"

    def get_unit(self) -> str:
        match self:
            case DataGroups.SurfaceAirTemperatureA | DataGroups.SurfaceAirTemperatureD:
                return "Kelvin (K)"
            case DataGroups.OzoneA | DataGroups.OzoneD:
                return "Dobson Unit (DU)"
            case (
                DataGroups.RelativeHumiditySurfaceA
                | DataGroups.RelativeHumiditySurfaceD
            ):
                return "%"

    def get_linear_regression(self):
        match self:
            case DataGroups.SurfaceAirTemperatureA | DataGroups.SurfaceAirTemperatureD:
                lr = LinearRegression()
                p = PolynomialFeatures(3)
                return lr, p
            case DataGroups.OzoneA | DataGroups.OzoneD:
                lr = LinearRegression()
                p = PolynomialFeatures(1)
                return lr, p
            case (
                DataGroups.RelativeHumiditySurfaceA
                | DataGroups.RelativeHumiditySurfaceD
            ):
                lr = LinearRegression()
                p = PolynomialFeatures(2)
                return lr, p

    def get_k_nearest_neighbours(self):
        match self:
            case DataGroups.SurfaceAirTemperatureA | DataGroups.SurfaceAirTemperatureD:
                knr = KNeighborsRegressor(n_neighbors=25)
                return knr
            case DataGroups.OzoneA | DataGroups.OzoneD:
                knr = KNeighborsRegressor(n_neighbors=100)
                return knr
            case (
                DataGroups.RelativeHumiditySurfaceA
                | DataGroups.RelativeHumiditySurfaceD
            ):
                knr = KNeighborsRegressor(n_neighbors=15)
                return knr

    def get_neural_network(self, features: int):
        print(f"Getting network for {self}")
        match self:
            case DataGroups.SurfaceAirTemperatureA | DataGroups.SurfaceAirTemperatureD:
                network = neural_network.MLP(
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
                epoch = 30
                lr = 0.01
                loss_function = nn.HuberLoss()
                return network, epoch, lr, loss_function
            case DataGroups.OzoneA | DataGroups.OzoneD:
                network = neural_network.MLP(
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
                epoch = 30
                lr = 0.01
                loss_function = nn.L1Loss()
                return network, epoch, lr, loss_function
            case (
                DataGroups.RelativeHumiditySurfaceA
                | DataGroups.RelativeHumiditySurfaceD
            ):
                network = neural_network.MLP(
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
                epoch = 40
                lr = 0.01
                loss_function = nn.L1Loss()
                return network, epoch, lr, loss_function

    def get_gaussian_process(self):
        match self:
            case DataGroups.SurfaceAirTemperatureA | DataGroups.SurfaceAirTemperatureD:
                epoch = 30
                lr = 0.1
                amount_inducing_points = 1000
                return epoch, lr, amount_inducing_points
            case DataGroups.OzoneA | DataGroups.OzoneD:
                epoch = 30
                lr = 0.1
                amount_inducing_points = 500
                return epoch, lr, amount_inducing_points
            case (
                DataGroups.RelativeHumiditySurfaceA
                | DataGroups.RelativeHumiditySurfaceD
            ):
                epoch = 30
                lr = 0.1
                amount_inducing_points = 500
                return epoch, lr, amount_inducing_points
