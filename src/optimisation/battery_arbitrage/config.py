from dataclasses import dataclass

import pandas as pd


@dataclass
class Request:
    timesteps: list[pd.Timestamp]
    current_state: float
    prices: dict[pd.Timestamp, float]
    battery_capacity: float
    max_power: float


@dataclass
class Config:
    battery_capacity: float
    max_power: float
