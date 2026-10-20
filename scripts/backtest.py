import pandas as pd

from src.data.raw import read_raw_data
from src.evaluation.backtesting import run_backtest, BacktestingConfig
from src.forecasting.models.base import ForecastVariables
from src.forecasting.models.rule_based import NaiveForecaster
from src.optimisation.battery_arbitrage.config import Config
from src.optimisation.battery_arbitrage.lp import BatteryArbitrageLP

if __name__ == "__main__":

    res = "hour"
    horizon_days = 1
    config = BacktestingConfig("2022-01-01", "2025-12-31", "2026-07-01", res)

    raw_data = read_raw_data(["day_ahead_price"], res)

    forecaster = NaiveForecaster(
        # "daily_naive", ForecastVariables(y="price"), params={"sp": 24}),
        "weekly_naive",
        ForecastVariables(y="price"),
        # todo: make this dynamic based on resolution and frequency of data
        params={"sp": 168},
    )
    optimiser = BatteryArbitrageLP

    backtest_results = run_backtest(
        raw_data,
        forecaster,
        optimiser,
        Config(battery_capacity=10.0, max_power=1.0),
        config,
    )
    print(pd.DataFrame(backtest_results))
