import pandas as pd

from src.data.raw import read_raw_data, Resolution, slice_dataframe
from src.evaluation.backtesting import run_backtest, BacktestingConfig

if __name__ == "__main__":

    config = BacktestingConfig(
        "2023-01-01", "2026-01-01", "2026-07-01", Resolution.HOURLY, "Europe/Berlin"
    )
    raw_data = read_raw_data(["day_ahead_price"], config.resolution)
    sliced_data = slice_dataframe(
        raw_data["day_ahead_price"],
        pd.Timestamp(config.fit_start, tz=config.timezone).tz_convert("UTC"),
        pd.Timestamp(config.end, tz=config.timezone).tz_convert("UTC"),
    )

    backtest_results = run_backtest(
        sliced_data,
        ["naive_weekly", "naive_daily", "perfect"],
        "battery_arbitrage_lp",
        "simple",
        config,
    )
    print(
        backtest_results.groupby(by="forecaster").agg(
            {
                "MAE": "mean",
                "RMSE": "mean",
                "pnl_forecast": "sum",
                "pnl_realised": "sum",
            }
        ).round(3)
    )
