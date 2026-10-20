from dataclasses import dataclass

import pandas as pd
from sktime.split import ExpandingWindowSplitter
from tqdm import tqdm

from src.data.raw import slice_dataframe
from src.forecasting.models.base import ForecastingModel
from src.optimisation.base import BaseORToolsLP
from src.optimisation.mpc import run_mpc


@dataclass
class BacktestingConfig:
    fit_start: str
    predict_start: str
    end: str
    resolution: str


def run_backtest(
    raw_data: pd.DataFrame,
    forecaster: ForecastingModel,
    optimiser: BaseORToolsLP,
    config: BacktestingConfig,
) -> dict:
    metrics = {}
    current_state = 0.0

    data, splitter = init_splitter(raw_data, config)

    for split in tqdm(
        splitter.split(data), desc=f"Backtesting {forecaster.name}", leave=True
    ):

        train = data.iloc[split[0]]
        test = data.iloc[split[1]]
        date = test.index[-1].date()

        metrics[date], current_state = run_mpc(
            forecaster, optimiser, train, test, current_state, date
        )

    return pd.DataFrame.from_dict(metrics, orient="index")


def init_splitter(
    all_data, config: BacktestingConfig
) -> tuple[pd.DataFrame, ExpandingWindowSplitter]:
    sliced_data = slice_dataframe(
        all_data["day_ahead_price"], config.fit_start, config.end
    )
    initial_train = len(
        slice_dataframe(sliced_data, config.fit_start, config.predict_start)
    )

    return sliced_data, ExpandingWindowSplitter(
        **get_split_params_from_resolution(config.resolution, initial_train + 12)
    )


def get_split_params_from_resolution(resolution: str, initial_window: int) -> dict:
    # todo: make initial_window dynamic??

    horizon_in_days = 1

    if resolution == "hour":
        horizon_hrs = 24 * horizon_in_days

        return {
            "fh": range(1, horizon_hrs + 12 + 1),
            "initial_window": initial_window,
            "step_length": horizon_hrs,
        }
    elif resolution == "quarterhour":
        horizon_qrt_hrs = 4 * 24 * horizon_in_days
        return {
            "fh": range(1, horizon_qrt_hrs + 12 * 4 + 1),
            "initial_window": initial_window,
            "step_length": horizon_qrt_hrs,
        }
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
