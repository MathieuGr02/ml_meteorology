import os
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xarray as xr
from enum import Enum
import json
import numpy as np

"""
This script provides methods which aid in the preparing of data through the creation of feature vectors etc.
"""


class DataGroups(Enum):
    SurfaceAirTemperatureA = "surface_air_temperature_A"
    SurfaceAirTemperatureD = "surface_air_temperature_D"


def get_group(group: DataGroups) -> tuple[list[str], str]:
    """
    Get training and target keys from a data group in the `data_groups.json` file
    """
    with open("data_groups.json", "r") as json_groups:
        json_groups = json.load(json_groups)
        group_keys = json_groups[group.value]
        return (group_keys["training"], group_keys["target"])


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
    i, j = np.meshgrid(np.arange(90, -90, -1), np.arange(-180, 180, 1), indexing="ij")
    feature_vector = np.column_stack((i.ravel(), j.ravel()))

    n, _ = feature_vector.shape

    variables = list(filter(lambda v: v not in ["Latitude", "Longitude"], data.dims))
    # Sometimes the DataArray is 3D, so get a 2D slice and add as feature to feature vector
    if len(variables) > 0:
        for variable in variables:
            for i in range(len(data[variable].values)):
                feature_vector = np.column_stack(
                    (feature_vector, data[i].values.ravel())
                )
    else:
        feature_vector = np.column_stack((feature_vector, data.values.ravel()))

    if not with_coordinates:
        feature_vector = feature_vector[:, 2:]

    return feature_vector


def drop_na(
    training: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Drop all NaN from both training and target data

    # Returns

    `training` and `target` without NaN values
    """
    target = target.ravel()
    mask = ~np.isnan(training).any(axis=1) & ~np.isnan(target)
    return training[mask], target[mask]


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
            if key == "LandSeaMask":
                subset.data[subset.data == 0] = np.nan
            subset_training_data.append(to_feature_vector(subset))

        subset = dataset[target]

        training_data.append(combine_feature_vectors(subset_training_data))
        target_data.append(to_feature_vector(subset, with_coordinates=False))

    return (
        np.array(training_data),
        np.array(target_data),
    )


def get_data(
    group: DataGroups,
) -> tuple[
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
]:
    """
    Get all data fully prepared.

    # Returns
    - training data
    - training target data
    - test data
    - test target data
    """
    training, target = get_group(group)

    # Get training and test data
    training_data, training_target_data = aggregate_data(
        training, target, file_keyword="2015"
    )
    test_data, test_target_data = aggregate_data(training, target, file_keyword="2016")

    n, *_ = test_data.shape

    new_training_data = []
    new_training_target_data = []
    new_test_data = []
    new_test_target_data = []

    scaler = StandardScaler()
    for i in range(n):
        training_data_wo_na, training_target_data_wo_na = drop_na(
            training_data[i], training_target_data[i]
        )
        test_data_wo_na, test_target_data_wo_na = drop_na(
            test_data[i], test_target_data[i]
        )

        # Scale data
        training_data_scaled = scaler.fit_transform(training_data_wo_na)
        test_data_scaled = scaler.fit_transform(test_data_wo_na)

        new_training_data.append(training_data_scaled)
        new_training_target_data.append(training_target_data_wo_na)
        new_test_data.append(test_data_scaled)
        new_test_target_data.append(test_target_data_wo_na)

    return (
        new_training_data,
        new_training_target_data,
        new_test_data,
        new_test_target_data,
    )
