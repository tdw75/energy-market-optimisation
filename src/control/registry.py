from src.control.base import controller, ControlConfig


def get_controller_by_name(name: str) -> controller:
    if name not in CONTROL_REGISTRY:
        raise ValueError(f"Policy '{name}' not found in the registry.")

    return CONTROL_REGISTRY[name]()


def simple_controller() -> controller:
    return controller(
        config=ControlConfig(step_length_days=1, forecast_horizon_days=1)
    )


def mpc_week_ahead() -> controller:
    return controller(
        config=ControlConfig(step_length_days=1, forecast_horizon_days=7)
    )


CONTROL_REGISTRY = {"simple": simple_controller, "mpc_week_ahead": mpc_week_ahead}
