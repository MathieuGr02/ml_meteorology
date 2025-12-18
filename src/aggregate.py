import json
import os
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xarray.core.utils import OrderedSet

from config import Config

"""
This script provides methods which aid in the preparing of data through the creation of feature vectors etc.
"""


def get_variables() -> tuple[list[str], list[str], list[str]]:
    """
    Get variables of the dataset

    # Returns
    All single meta variables (1) and multiple variables which have a A and D variable (2, 3)
    - returns (1), (2), (3)
    """
    with open("resources/variables.json", "r") as variables:
        json_file = json.load(variables)
        single_variables = json_file["single_variables"]
        multi_variables = json_file["multi_variables"]

        multi_variables_A = [f"{var}_A" for var in multi_variables]
        multi_variables_D = [f"{var}_D" for var in multi_variables]

        return single_variables, multi_variables_A, multi_variables_D


def combine_feature_vectors(data):
    """
    Combine the different features of two vectors into one. Assumption is that the first two columns are Longitude and Latitude.
    """
    # Add (Longitude, Latitude) as first 2 columns of feature vector
    feature_vectors = (data[0][:, :2],)
    for vector in data:
        # Add features without (Longitude, Latitude)
        feature_vectors += (vector[:, 2:],)

    return np.column_stack(feature_vectors)


def to_feature_vector(
    data: xr.DataArray, with_coordinates: bool = True
) -> npt.NDArray[np.float64]:
    """
    Trainsform a xarray DataArray to a feature vector. The Longitude and Latitude coordinates are mapped to the first columns of the feature vector.

    # Returns
    Numpy array:
    - With coordinates
    [Longitude, Latitude, <rest of the features>]
    - Without coordinates
    [<rest of features>]
    """

    lat = data.Latitude.values
    lon = data.Longitude.values

    lon2d, lat2d = np.meshgrid(lon, lat)

    if len(data.values.shape) == 3:
        tuple = ()
        for v in data.values:
            tuple += (v.reshape(-1, 1),)
        features = np.column_stack(tuple)
    else:
        features = data.values.reshape(-1, 1)

    if with_coordinates:
        feature_vector = np.column_stack((lon2d.ravel(), lat2d.ravel(), features))
    else:
        feature_vector = features

    return feature_vector


def drop_na(
    training: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Drop all NaN from both training and target data

    # Returns

    `training` and `target` without NaN values
    """
    # target = target.ravel()
    mask = ~np.isnan(training).any(axis=1) & ~np.isnan(target).any(axis=1)
    return training[mask], target[mask], mask


def aggregate_data(
    training: list[str], target: str, file_keyword: str | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    training: List of training keys
    target: List of target keys

    # Returns

    Feature vectors of training and target data
    """

    training_data = []
    target_data = []

    for file in os.listdir("./data"):
        if file_keyword is not None and not file.__contains__(file_keyword):
            continue

        dataset = xr.open_dataset(f"./data/{file}")

        subset_training_data = []

        # Create feature vector out of training keys
        for key in training:
            subset = dataset[key]
            subset_training_data.append(to_feature_vector(subset))

        subset = dataset[target]

        training_data.append(combine_feature_vectors(subset_training_data))
        target_data.append(to_feature_vector(subset, with_coordinates=False))

    return (
        np.array(training_data),
        np.array(target_data),
    )


def get_data(config: Config, file_keyword: str | list[str]):
    """
    Get data

    # Returns
    - 3D array with (days x data points in grid x features)
    """
    keys = OrderedSet(["Longitude", "Latitude"])
    lon_low, lon_high = config.lon_low + 180, config.lon_high + 180
    lat_low, lat_high = (
        90 - config.lat_high,
        90 - config.lat_low,
    )

    print(f"Getting data. Filtering to files containing {file_keyword}")

    single, multi_A, multi_D = get_variables()

    files = os.listdir("./newdata")

    if file_keyword is not None:
        if type(file_keyword) is list:
            new_files = []
            for keyword in file_keyword:
                new_files.extend(filter(lambda f: f.__contains__(keyword), files))
            files = new_files
        else:
            files = filter(lambda f: f.__contains__(file_keyword), files)

    files = sorted(
        files,
        key=lambda s: datetime.strptime(
            s.split(".")[1] + "." + s.split(".")[2] + "." + s.split(".")[3], "%Y.%m.%d"
        ),
    )

    cube = []

    for file in files:
        print(f"Reading {file}")
        dataset = xr.open_dataset(f"./newdata/{file}")

        # Iterate over A and D variables
        # Process A and then D of the same day to add them after eachother in the matrix to represent 12 h skips
        for multi in [multi_A, multi_D]:
            subcube = []
            # Iterate over all feature keys
            for key in single + multi:
                subset = dataset[key]

                # Index grid scope
                if len(subset.shape) == 3:
                    subset = subset[:, lat_low:lat_high, lon_low:lon_high]
                else:
                    subset = subset[lat_low:lat_high, lon_low:lon_high]

                # Add key to keys list, and if key has multiple subkeys, add all subkeys
                if len(subset.dims) == 2:
                    transformed_key = key
                    if "_A" in key:
                        transformed_key = key.replace("_A", "")
                    if "_D" in key:
                        transformed_key = key.replace("_D", "")
                    keys.add(transformed_key)
                else:
                    for dim in list(subset.dims):
                        if dim not in keys:
                            for value in subset[dim].values:
                                transformed_key = key
                                if "_A" in key:
                                    transformed_key = key.replace("_A", "")
                                if "_D" in key:
                                    transformed_key = key.replace("_D", "")
                                keys.add(f"{transformed_key}-{dim}-{value}")

                features = to_feature_vector(subset)
                subcube.append(features)

            # Stack different key feature vectors into one
            # Key: x has N x k and Key: y has N x l
            # => output N x (k + l - 2) (-2 because of the longitude and latitude which do not get duplicated)
            # Add the feature vector of the day A / D to the cube s.t. it has a shape (2 * #days, #samples, #features)
            feature_vectors = combine_feature_vectors(subcube)

            day_time_feature = np.empty(feature_vectors.shape[0])
            if multi == multi_A:
                day_time_feature.fill(1)
            else:
                day_time_feature.fill(-1)

            # feature_vectors = np.column_stack((feature_vectors, day_time_feature))
            # keys.add("Time")

            cube.append(feature_vectors)

    cube = np.array(cube)

    imp = SimpleImputer(strategy="mean")

    for i in range(cube.shape[0]):
        cube[i] = imp.fit_transform(cube[i])

    return cube, keys


def get_time_data(config: Config, file_keyword: str | list[str]):
    print("Getting time data")
    cube, keys = get_data(config, file_keyword)

    print(cube.shape)

    steps, samples, features = cube.shape

    data = None
    targets = None
    print("Creating time series data")
    for i in range(config.lag, steps - config.leads):
        lag_subset = cube[i - config.lag + 1 : i + 1]
        lead_subset = cube[i + 1 : i + config.leads + 1, :, 4:]

        X = lag_subset.transpose(1, 0, 2)
        n, t, f = X.shape
        X = X.reshape(n, f * t)

        y = lead_subset.transpose(1, 0, 2)
        n, t, f = y.shape
        y = y.reshape(n, f * t)

        if data is None:
            data = X
        else:
            data = np.vstack((data, X))

        if targets is None:
            targets = y
        else:
            targets = np.vstack((targets, y))

    data, targets = np.array(data), np.array(targets)

    return data, targets, list(keys)[4:], cube.shape
