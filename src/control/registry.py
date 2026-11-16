from src.control.base import Controller, ControlConfig


def get_controller_by_name(name: str) -> Controller:
    if name not in CONTROL_REGISTRY:
        raise ValueError(f"Policy '{name}' not found in the registry.")

    return CONTROL_REGISTRY[name]()


def simple_controller() -> Controller:
    return Controller(
        config=ControlConfig(step_length_days=1, forecast_horizon_days=1)
    )


def mpc_week_ahead() -> Controller:
    return Controller(
        config=ControlConfig(step_length_days=1, forecast_horizon_days=7)
    )


CONTROL_REGISTRY = {"simple": simple_controller, "mpc_week_ahead": mpc_week_ahead}
