from dataclasses import dataclass


@dataclass
class Config:
    lon_low: int
    lon_high: int
    lat_low: int
    lat_high: int
    lag: int = 14  # Amount of steps used to train, 7 days in 12 h steps
    leads: int = 1  # Amount of steps ahead to predict
    train_year = "2015"
    test_year = "2016"
