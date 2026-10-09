import pandas as pd
from sktime.split import ExpandingWindowSplitter

from src.data.raw import read_raw_data, slice_dataframe
from src.evaluation.backtesting import run_backtest
from src.forecasting.models.base import NaiveForecaster, ForecastVariables


def get_split_params_from_resolution(resolution: str, initial_window: int) -> dict:
    # todo: make initial_window dynamic??

    if resolution == "hour":
        return {"fh": range(1, 37), "initial_window": initial_window, "step_length": 24}
    elif resolution == "quarterhour":
        return {
            "fh": range(1, 145),
            "initial_window": initial_window,
            "step_length": 96,
        }
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


if __name__ == "__main__":

    fit_start = "2022-01-01"
    predict_start = "2026-01-01"
    end = "2026-07-01"
    res = "hour"

    all_data = read_raw_data(["day_ahead_price"], res)
    data = slice_dataframe(all_data["day_ahead_price"], fit_start, end)
    initial_train = len(slice_dataframe(data, fit_start, predict_start))

    splitter = ExpandingWindowSplitter(
        **get_split_params_from_resolution(res, initial_train + 12)
    )
    backtest_results = run_backtest(
        data,
        NaiveForecaster(
            # "daily_naive", ForecastVariables(y="price"), params={"sp": 24}),
            "weekly_naive",
            ForecastVariables(y="price"),
            params={
                "sp": 168
            },  # todo: make this dynamic based on resolution and frequency of data
        ),
        splitter,
    )
    print(pd.DataFrame(backtest_results).mean(axis=1))
