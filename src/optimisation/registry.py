from src.optimisation.battery_arbitrage.config import BatteryArbitrageConfig
from src.optimisation.battery_arbitrage.lp import BatteryArbitrageLP


def get_optimiser_by_name(name: str, current_state=None, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Optimiser '{name}' not found in the registry.")

    return MODEL_REGISTRY[name](current_state, **kwargs)


def battery_arbitrage_lp(current_state: float, **kwargs):
    default = {"battery_capacity": 10.0, "max_power": 1.0}
    default.update(kwargs)
    config = BatteryArbitrageConfig(**default)
    return BatteryArbitrageLP(config, current_state=current_state or 0.0)


MODEL_REGISTRY = {"battery_arbitrage_lp": battery_arbitrage_lp}
