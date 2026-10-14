from dataclasses import dataclass


@dataclass
class Request:
    timesteps: list[int]
    current_state: float
    prices: dict[int, float]
    battery_capacity: float
    max_power: float


@dataclass
class Config:
    battery_capacity: float
    max_power: float
