import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.evaluation.backtesting.config import BacktestingConfig
from src.evaluation.metrics.forecasting import evaluate_forecast_accuracy
from src.evaluation.metrics.pnl import evaluate_pnl
from src.optimisation.battery_arbitrage.config import BatteryConfig
from src.optimisation.battery_arbitrage.lp import schedule_schema
from src.utils import get_datetime_index_schema_checks

# todo: convert timezone to UTC when storing results, and convert back to local timezone when loading for evaluation??


@dataclass
class BacktestResults:
    metadata: BacktestingConfig
    battery: BatteryConfig

    price: pd.DataFrame
    schedule: pd.DataFrame
    state: pd.Series
    metrics: pd.DataFrame

    def store_iteration(
        self,
        forecasts: pd.Series,
        actual: pd.DataFrame,
        schedule: pd.DataFrame,
        state: float,
        date: pd.Timestamp,
    ):
        self._store_prices(
            forecasts.reindex(schedule.index), actual.reindex(schedule.index)
        )
        self._store_schedule(schedule)
        self._store_state(state, date)

    def _store_prices(self, forecasts: pd.Series, actual: pd.DataFrame):
        assert forecasts.index.equals(
            actual.index
        ), "Forecasts and actual data must have the same index."
        prices = pd.DataFrame({"actual": actual["price"], "pred": forecasts})
        prices = price_schema().validate(prices)
        self.price = pd.concat([self.price, prices], axis=0)

    def _store_schedule(self, schedule: pd.DataFrame):
        schedule = schedule_schema(self.battery).validate(schedule)
        self.schedule = pd.concat([self.schedule, schedule], axis=0)

    def _store_state(self, state: float, date: pd.Timestamp):
        self.state.loc[date] = state

    def evaluate_iteration(
        self,
        executed_schedule: pd.DataFrame,
        pred: pd.Series,
        actual: pd.DataFrame,
        date: pd.Timestamp,
    ):

        actual_reindexed = actual.reindex(executed_schedule.index)["price"]
        pred_reindexed = pred.reindex(executed_schedule.index)

        forecast_metrics = evaluate_forecast_accuracy(actual_reindexed, pred_reindexed)
        pnl_metrics = evaluate_pnl(executed_schedule, actual_reindexed, pred_reindexed)

        metrics = {**forecast_metrics, **pnl_metrics.sum().to_dict()}
        row = pd.DataFrame(metrics, index=[date])
        self.metrics = pd.concat([self.metrics, row], axis=0)

    def validate_final(self):
        self.price = price_schema().validate(self.price)
        self.schedule = schedule_schema(self.battery).validate(self.schedule)
        self.state = state_schema().validate(self.state)
        self.metrics = metrics_schema().validate(self.metrics)

    def save_to_file(self, directory: str | Path):

        self.validate_final()

        directory_path = (
            Path(directory) / self.metadata.backtest_id / self.metadata.run_id
        )
        directory_path.mkdir(parents=True, exist_ok=True)

        with open(directory_path / "config.json", "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=4)
        with open(directory_path / "battery.json", "w") as f:
            json.dump(self.battery.__dict__, f, indent=4)
        self.price.to_csv(directory_path / "prices.csv")
        self.schedule.to_csv(directory_path / "schedule.csv")
        self.state.to_csv(directory_path / "state.csv")
        self.metrics.to_csv(directory_path / "metrics.csv")


def price_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "actual": pa.Column(float, nullable=False),
            "pred": pa.Column(float, nullable=False),
        },
        checks=get_datetime_index_schema_checks(),
    )


def metrics_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "mae": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
            "rmse": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
            "mhd": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
            "mpd": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
            "pnl_forecast": pa.Column(float, nullable=False),
            "pnl_realised": pa.Column(float, nullable=False),
        },
        checks=get_datetime_index_schema_checks(),
    )


def state_schema() -> pa.SeriesSchema:
    return pa.SeriesSchema(
        float,
        nullable=False,
        checks=get_datetime_index_schema_checks(),
        # name="state",  # todo: add name to series schema when defining
    )
