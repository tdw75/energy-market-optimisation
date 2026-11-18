from src.data.raw import Resolution
from src.optimisation.battery_arbitrage.config import BatteryConfig
from src.optimisation.battery_arbitrage.lp import BatteryArbitrageLP


def get_optimiser_by_name(
    name: str, resolution: Resolution, current_state=None, **kwargs
):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Optimiser '{name}' not found in the registry.")

    return MODEL_REGISTRY[name](resolution, current_state, **kwargs)


def battery_arbitrage_lp(resolution: Resolution, current_state: float, **kwargs):
    default = {
        "battery_capacity": 10.0,
        "max_power": 1.0,
        "charging_efficiency": 1.0,
        "discharging_efficiency": 1.0,
    }
    default.update(kwargs)
    config = BatteryConfig(**default)
    return BatteryArbitrageLP(
        config, current_state=current_state or 0.0, resolution=resolution
    )


MODEL_REGISTRY = {"battery_arbitrage_lp": battery_arbitrage_lp}
