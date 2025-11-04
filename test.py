import xarray as xr

if __name__ == "__main__":
    file = "AIRS.2015.03.31.L3.RetStd001.v7.0.3.0.G20177162716.hdf.nc4"
    dataset = xr.open_dataset(f"./rawdata/{file}")

    print(list(dataset.data_vars))

    print(dataset["Temperature_A"])
