from src.data.raw import Resolution
from src.forecasting.models.base import ForecastVariables, ForecastingModel
from src.forecasting.models.foundational import Timestral5MForecaster
from src.forecasting.models.linear import SARIMAForecaster
from src.forecasting.models.neural import DLinearForecaster, NLinearForecaster
from src.forecasting.models.rule_based import (
    PerfectForecaster,
    WeeklyNaiveForecaster,
    DailyNaiveForecaster,
)
from src.forecasting.models.tree import (
    BoostedForecasterWithWindow,
    BoostedForecasterWithLags,
)


def get_forecaster_by_name(name: str, resolution: Resolution) -> ForecastingModel:

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in the registry.")

    # todo: make this dynamic based on the model and data
    variables = ForecastVariables(y="price")

    return MODEL_REGISTRY[name](variables, resolution)


def perfect_forecaster(variables: ForecastVariables, resolution: Resolution):
    return PerfectForecaster("perfect", variables, resolution)


def naive_weekly(variables: ForecastVariables, resolution: Resolution):
    return WeeklyNaiveForecaster("naive_weekly", variables, resolution)


def naive_daily(variables: ForecastVariables, resolution: Resolution):
    return DailyNaiveForecaster("naive_daily", variables, resolution)


def dlinear_forecaster(variables: ForecastVariables, resolution: Resolution):
    return DLinearForecaster("dlinear", variables, resolution)


def nlinear_forecaster(variables: ForecastVariables, resolution: Resolution):
    return NLinearForecaster("nlinear", variables, resolution)


def timestral_5m_forecaster(variables: ForecastVariables, resolution: Resolution):
    return Timestral5MForecaster("timestral", variables, resolution)


def catboost_manual_forecaster(variables: ForecastVariables, resolution: Resolution):
    return BoostedForecasterWithLags("catboost_manual", variables, resolution)


def catboost_forecaster(variables: ForecastVariables, resolution: Resolution):
    return BoostedForecasterWithWindow("catboost", variables, resolution)


def sarima_forecaster(variables: ForecastVariables, resolution: Resolution):
    return SARIMAForecaster("sarima", variables, resolution)


MODEL_REGISTRY = {
    "perfect": perfect_forecaster,
    "naive_weekly": naive_weekly,
    "naive_daily": naive_daily,
    "dlinear": dlinear_forecaster,
    "nlinear": nlinear_forecaster,
    "timestral_5m": timestral_5m_forecaster,
    "catboost_manual": catboost_manual_forecaster,
    "catboost": catboost_forecaster,
    "sarima": sarima_forecaster,
}
