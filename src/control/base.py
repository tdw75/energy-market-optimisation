from dataclasses import dataclass
from typing import Any

import numpy as np
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
        execution_date: pd.Timestamp,
    ) -> tuple[pd.Series, pd.DataFrame, BaseORToolsLP]:
        forecasts = forecaster.fit_predict(train, test)
        optimise_horizon = self._get_forecast_horizon(forecasts, execution_date)
        schedule = optimiser.solve(forecasts[optimise_horizon])
        return forecasts, schedule, optimiser

    def execute(
        self,
        optimiser: BaseORToolsLP,
        schedule: pd.DataFrame,
        execution_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, Any]:
        horizon = self._get_schedule_horizon(schedule, execution_date)
        executed_schedule = schedule.loc[horizon]
        state = optimiser.update_state(executed_schedule)
        return executed_schedule, state

    def _get_forecast_horizon(
        self, forecasts: pd.Series | pd.DataFrame, start: pd.Timestamp
    ) -> np.ndarray:
        return self._get_horizon(forecasts, start, self.config.forecast_horizon_days)

    def _get_schedule_horizon(
        self, schedule: pd.Series | pd.DataFrame, start: pd.Timestamp
    ) -> np.ndarray:
        return self._get_horizon(schedule, start, self.config.step_length_days)

    @staticmethod
    def _get_horizon(
        data: pd.Series | pd.DataFrame, start: pd.Timestamp, offset_days: int
    ) -> np.ndarray:
        end = start + pd.Timedelta(days=offset_days)
        return (data.index >= start) & (data.index < end)
