# ml_meteorology

Prediction of metereological data using different machine learning methods. Project for the 2025HS Scientific Writing lecture

# Data

The data comes from [https://disc.gsfc.nasa.gov/datasets/AIRX3STD_7.0/summary](https://disc.gsfc.nasa.gov/datasets/AIRX3STD_7.0/summary) and can be downloaded publicly via the `download.py` script. Note that for this, an `.env` file with `TOKEN=<GES-DISC-ACCOUNT-TOKEN>` is required.

# Files

All code is found in the `<name>.py` files. The `keys.json` contains all the variables of the data after applying the filtering with `removable_keys.json` on the raw data. The `data_groups.json` contains the analysis groups with the features and the targets.
