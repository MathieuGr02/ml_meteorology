import itertools
from aggregate import DataGroups, get_data, get_time_data
from tqdm import tqdm
from regression import train, degree_fit, predict
from cuml.linear_model import LinearRegression
from cuml.preprocessing import PolynomialFeatures
from cuml.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    group = DataGroups.SurfaceAirTemperatureA

    lon_low = -20  # 170
    lon_high = 60  # 210
    lat_low = 30  # 120
    lat_high = 80  # 160

    X_train, y_train, keys, original_shape = get_time_data(
        group,
        "2015",
        lag=7,
        lon_low=lon_low,
        lon_high=lon_high,
        lat_low=lat_low,
        lat_high=lat_high,
    )
    X_test, y_test, keys, original_shape = get_time_data(
        group,
        "2016",
        lag=7,
        lon_low=lon_low,
        lon_high=lon_high,
        lat_low=lat_low,
        lat_high=lat_high,
    )
    print(keys)
    features = y_test.shape[1]

    print(X_train.shape)

    p = PolynomialFeatures(2)
    lr = LinearRegression()

    X_train_p, X_test_p = degree_fit(X_train, X_test, p)
    train(lr, X_train_p, y_train)
    error, output = predict(lr, X_test_p, y_test)

    print(error, y_test.shape)

    test_reshaped = y_test.reshape(24, lat_high - lat_low, lon_high - lon_low, features)
    output_reshaped = output.reshape(
        24, lat_high - lat_low, lon_high - lon_low, features
    )

    key_index = keys.index("SurfAirTemp_A")

    key_error = mean_absolute_error(y_test[:, key_index], output[:, key_index])
    print(f"Surface Air Temperature A error: {key_error}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    pc1 = ax1.pcolormesh(np.flipud(test_reshaped[1, :, :, key_index]))
    pc2 = ax2.pcolormesh(np.flipud(output_reshaped[1, :, :, key_index]))
    fig.colorbar(pc1)
    fig.colorbar(pc2)
    plt.savefig("testing.png")
