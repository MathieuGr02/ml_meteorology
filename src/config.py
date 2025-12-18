from dataclasses import dataclass


@dataclass
class Config:
    lon_low: int
    lon_high: int
    lat_low: int
    lat_high: int
    lag: int = 8  # Amount of steps used to train (1 step = 12 h)
    leads: int = 1  # Amount of steps ahead to predict (1 step = 12 h)
    train_year = "2014"
    test_year = "2015"
    months = ["07", "08", "09"]
