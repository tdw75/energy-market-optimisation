from src.data.raw import Resolution
from src.forecasting.models.base import ForecastVariables
from src.forecasting.models.rule_based import PerfectForecaster, NaiveForecaster


def get_forecaster_by_name(
    name: str, resolution: Resolution
) -> PerfectForecaster | NaiveForecaster:

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in the registry.")

    # todo: make this dynamic based on the model and data
    variables = ForecastVariables(y="price")

    return MODEL_REGISTRY[name](variables, resolution)


def perfect_forecaster(variables: ForecastVariables, resolution: Resolution):
    return PerfectForecaster("perfect", variables)


def naive_weekly(variables: ForecastVariables, resolution: Resolution):
    params = {"sp": resolution.steps_per_week}
    return NaiveForecaster("naive_weekly", variables, params)


def naive_daily(variables: ForecastVariables, resolution: Resolution):
    params = {"sp": resolution.steps_per_day}
    return NaiveForecaster("naive_daily", variables, params)


MODEL_REGISTRY = {
    "perfect": perfect_forecaster,
    "naive_weekly": naive_weekly,
    "naive_daily": naive_daily,
}
