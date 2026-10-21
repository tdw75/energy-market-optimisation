from src.forecasting.models.base import ForecastVariables
from src.forecasting.models.rule_based import PerfectForecaster, NaiveForecaster


def get_forecaster_by_name(name: str):

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in the registry.")

    # todo: make this dynamic based on the model and data
    variables = ForecastVariables(y="price")

    return MODEL_REGISTRY[name](variables)


def perfect_forecaster(variables: ForecastVariables):
    return PerfectForecaster("perfect", variables)


def naive_weekly(variables: ForecastVariables, resolution: str = "hour"):
    sp = {"hour": 7 * 24, "quarterhour": 7 * 24 * 4}
    params = {"sp": sp.get(resolution, 7 * 24)}
    return NaiveForecaster("naive_weekly", variables, params)


def naive_daily(variables: ForecastVariables, resolution: str = "hour"):
    sp = {"hour": 24, "quarterhour": 24 * 4}
    params = {"sp": sp.get(resolution, 24)}
    return NaiveForecaster("naive_daily", variables, params)


MODEL_REGISTRY = {
    "perfect": perfect_forecaster,
    "naive_weekly": naive_weekly,
    "naive_daily": naive_daily,
}
