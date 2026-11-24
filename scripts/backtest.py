from pathlib import Path

import pandas as pd

from src.data.raw import read_raw_data, Resolution, slice_dataframe
from src.evaluation.backtesting.backtesting import run_backtest
from src.evaluation.backtesting.config import BacktestingConfig, generate_backtest_id
from src.optimisation.battery_arbitrage.config import BatteryConfig


def summarise_results(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics_agg = metrics.agg(
        {
            "mae": "mean",
            "rmse": "mean",
            "mhd": "mean",
            "pnl_forecast": "sum",
            "pnl_realised": "sum",
        }
    )

    return metrics_agg.round(3).T


if __name__ == "__main__":

    # todo: move this to a config yaml or command line arguments
    directory = Path(__file__).parent.parent / "resources" / "backtests"
    run_name = "backtest"
    forecasters = ["naive_daily", "naive_weekly", "perfect"]
    optimiser_name = "battery_arbitrage_lp"
    controller_name = "mpc_week_ahead"
    battery = {
        "battery_capacity": 10.0,
        "max_power": 1.0,
        "charging_efficiency": 1.0,
        "discharging_efficiency": 1.0,
    }
    backtest_id = generate_backtest_id(run_name)
    summary = {}

    for idx, forecaster_name in enumerate(forecasters):

        config = BacktestingConfig(
            backtest_id=backtest_id,
            run_idx=idx,
            fit_start="2023-01-01",
            opt_start="2026-01-01",
            end="2026-07-01",
            resolution=Resolution.HOURLY,
            forecaster_name=forecaster_name,
            optimiser_name=optimiser_name,
            controller_name=controller_name,
            timezone="Europe/Berlin",
        )
        raw_data = read_raw_data(["day_ahead_price"], config.resolution)
        sliced_data = slice_dataframe(
            raw_data["day_ahead_price"],
            pd.Timestamp(config.fit_start, tz=config.timezone).tz_convert("UTC"),
            pd.Timestamp(config.end, tz=config.timezone).tz_convert("UTC"),
        )
        results = run_backtest(sliced_data, BatteryConfig(**battery), config)
        results.save_to_file(directory)

        summary[forecaster_name] = summarise_results(results.metrics)

    pd.DataFrame.from_dict(summary, orient="index").to_csv(
        directory / backtest_id / "summary.csv"
    )
