import pandas as pd

from src.data.raw import read_raw_data, Resolution, slice_dataframe
from src.evaluation.backtesting.backtesting import run_backtest
from src.evaluation.backtesting.config import BacktestingConfig
from src.optimisation.battery_arbitrage.config import BatteryConfig


def summarise_results(metrics: pd.DataFrame):
    return (
        metrics.agg(
            {
                "mae": "mean",
                "rmse": "mean",
                "mhd": "mean",
                "pnl_forecast": "sum",
                "pnl_realised": "sum",
            }
        ).round(3).T
    )


if __name__ == "__main__":
    results = {}
    forecasters = ["naive_weekly", "naive_daily", "perfect"]
    battery = {
        "battery_capacity": 10.0,
        "max_power": 1.0,
        "charging_efficiency": 1.0,
        "discharging_efficiency": 1.0,
    }

    for forecaster_name in forecasters:
        config = BacktestingConfig(
            run_id="run_id",
            fit_start="2023-01-01",
            opt_start="2026-01-01",
            end="2026-07-01",
            resolution=Resolution.HOURLY,
            forecaster_name=forecaster_name,
            optimiser_name="battery_arbitrage_lp",
            controller_name="mpc_week_ahead",
            timezone="Europe/Berlin",
        )
        raw_data = read_raw_data(["day_ahead_price"], config.resolution)
        sliced_data = slice_dataframe(
            raw_data["day_ahead_price"],
            pd.Timestamp(config.fit_start, tz=config.timezone).tz_convert("UTC"),
            pd.Timestamp(config.end, tz=config.timezone).tz_convert("UTC"),
        )
        results[forecaster_name] = run_backtest(
            sliced_data, BatteryConfig(**battery), config
        )
        print(f"Results for forecaster: {forecaster_name}")
        print("Metrics:")
        print(summarise_results(results[forecaster_name].metrics))