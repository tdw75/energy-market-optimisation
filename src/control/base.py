from abc import ABC
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.forecasting.models.base import ForecastingModel
from src.optimisation.base import BaseORToolsLP


@dataclass
class ControlConfig:
    step_length_days: int
    forecast_horizon_days: int


class Controller:

    def __init__(self, config: ControlConfig):
        self.config = config

    def plan(
        self,
        forecaster: ForecastingModel,
        optimiser: BaseORToolsLP,
        train: pd.DataFrame,
        test: pd.DataFrame,
        date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, BaseORToolsLP]:
        forecasts = forecaster.fit_predict(train, test)
        optimise_horizon = self._get_horizon(
            forecasts, date, self.config.forecast_horizon_days
        )
        schedule = optimiser.solve(forecasts[optimise_horizon])
        return forecasts, schedule, optimiser

    def execute(
        self,
        optimiser: BaseORToolsLP,
        schedule: pd.DataFrame,
        date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, Any]:
        horizon = self._get_horizon(schedule, date, self.config.step_length_days)
        executed_schedule = schedule[horizon]
        state = optimiser.update_state(executed_schedule)
        return executed_schedule, state

    @staticmethod
    def _get_horizon(
            data: pd.DataFrame, start: pd.Timestamp, offset_days: int
    ) -> pd.DatetimeIndex:
        end = start + pd.Timedelta(days=offset_days)
        return (data.index.date >= start) & (data.index.date < end)
