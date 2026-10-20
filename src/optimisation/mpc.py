from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.accuracy import evaluate_forecast_accuracy
from src.forecasting.models.base import ForecastingModel
from src.optimisation.base import BaseORToolsLP
from src.optimisation.battery_arbitrage.evaluation import evaluate_pnl


@dataclass
class MPCConfig:
    horizon: int
    timestep: int


def run_mpc(
    forecaster: ForecastingModel,
    optimiser: BaseORToolsLP,
    train: pd.DataFrame,
    test: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[dict, Any]:
    forecast = forecaster.fit_predict(train, test)
    forecast_metrics = evaluate_forecast_accuracy(forecast, test)

    # todo: optimise on whole horizon, minus today's forecasts
    schedule = optimiser.solve(forecast[forecast.index.date == date])
    state = optimiser.update_state(schedule)
    pnl_metrics = evaluate_pnl(
        schedule, test[test.index.date == date], forecast[forecast.index.date == date]
    )
    # todo: evaluate only on actioned date (tomorrow's schedule)
    metrics = {**forecast_metrics, **pnl_metrics.sum().to_dict()}
    return metrics, state
