from dataclasses import dataclass

import pandas as pd


@dataclass
class BatteryArbitrageRequest:
    timesteps: list[pd.Timestamp]
    current_state: float
    prices: dict[pd.Timestamp, float]
    battery_capacity: float
    max_power: float


@dataclass
class BatteryArbitrageConfig:
    battery_capacity: float
    max_power: float
