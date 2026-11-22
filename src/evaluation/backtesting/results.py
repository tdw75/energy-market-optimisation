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
    config: BacktestingConfig
    battery: BatteryConfig

    price: pd.DataFrame
    schedule: pd.DataFrame
    state: pd.Series
    metrics: pd.DataFrame

    def store_prices(self, forecasts: pd.Series, actual: pd.DataFrame):
        assert forecasts.index.equals(
            actual.index
        ), "Forecasts and actual data must have the same index."
        prices = pd.DataFrame({"actual": actual["price"], "pred": forecasts})
        prices = price_schema().validate(prices)
        self.price = pd.concat([self.price, prices], axis=0)

    def store_schedule(self, schedule: pd.DataFrame):
        schedule = schedule_schema(self.battery).validate(schedule)
        self.schedule = pd.concat([self.schedule, schedule], axis=0)

    def store_state(self, state: float, date: pd.Timestamp):
        self.state.loc[date] = state

    def evaluate_iteration(
        self, schedule: pd.DataFrame, pred: pd.Series, actual: pd.DataFrame, date
    ):
        forecast_metrics = evaluate_forecast_accuracy(actual, pred)
        pnl_metrics = evaluate_pnl(
            schedule,
            actual.reindex(schedule.index),
            pred.reindex(schedule.index),
        )
        metrics = {**forecast_metrics, **pnl_metrics.sum().to_dict()}
        row = pd.DataFrame(metrics, index=[date])
        self.metrics = pd.concat([self.metrics, row], axis=0)


    def save_to_file(self, directory: str):

        # validate
        self.price = price_schema().validate(self.price)
        self.schedule = schedule_schema(self.battery).validate(self.schedule)
        self.state = state_schema().validate(self.state)
        self.metrics = metrics_schema().validate(self.metrics)

        # save to file
        directory_path = Path(directory) / self.config.run_id
        directory_path.parent.mkdir(parents=True, exist_ok=True)

        with open(f"{directory_path}/config.txt", "w") as f:
            json.dump(self.config.__dict__, f, indent=4)
        with open(f"{directory_path}/battery.txt", "w") as f:
            json.dump(self.battery.__dict__, f, indent=4)
        self.price.to_csv(f"{directory_path}/prices.csv")
        self.schedule.to_csv(f"{directory_path}/schedule.csv")
        self.state.to_csv(f"{directory_path}/state.csv")
        self.metrics.to_csv(f"{directory_path}/metrics.csv")

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
        name="state",
    )
