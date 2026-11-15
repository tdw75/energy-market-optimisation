import pandas as pd

from src.data.raw import read_raw_data
from src.evaluation.backtesting import run_backtest, BacktestingConfig

if __name__ == "__main__":

    res = "hour"
    horizon_days = 1
    config = BacktestingConfig("2022-01-01", "2025-12-31", "2026-07-01", res)
    raw_data = read_raw_data(["day_ahead_price"], res)

    backtest_results = run_backtest(
        raw_data["day_ahead_price"],
        ["naive_weekly", "naive_daily", "perfect"],
        "battery_arbitrage_lp",
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
        )
    )
