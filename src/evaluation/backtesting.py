from dataclasses import dataclass

import pandas as pd
from sktime.split import ExpandingWindowSplitter
from tqdm import tqdm

from src.data.raw import slice_dataframe
from src.forecasting.models.base import ForecastingModel
from src.optimisation.base import BaseORToolsLP
from src.optimisation.battery_arbitrage.config import Config
from src.optimisation.mpc import run_mpc


@dataclass
class BacktestingConfig:
    fit_start: str
    opt_start: str
    end: str
    resolution: str
    step_length_days: int = 1  # for MPC
    forecast_horizon_days: int = 7  # for MPC


def run_backtest(
    raw_data: pd.DataFrame,
    forecaster: ForecastingModel,
    optimiser_cls: type[BaseORToolsLP],
    opt_config: Config,  # todo: rename
    config: BacktestingConfig,
) -> dict:
    metrics = {}
    state = 0.0

    data, splitter = init_splitter(raw_data, config)

    for split in tqdm(
        splitter.split(data), desc=f"Backtesting {forecaster.name}", leave=True
    ):

        # todo: add multiple forecasting models for each split

        train = data.iloc[split[0]]
        test = data.iloc[split[1]]
        date = test.index[
            splitter.step_length
        ].date()  # todo: test that this is correct
        optimiser = optimiser_cls(opt_config, state)
        metrics[date], state = run_mpc(forecaster, optimiser, train, test, date)

        # todo: summarise metrics by forecasting model
        # todo: add option to save metrics to csv

    return pd.DataFrame.from_dict(metrics, orient="index")


def init_splitter(
    all_data, config: BacktestingConfig
) -> tuple[pd.DataFrame, ExpandingWindowSplitter]:
    sliced_data = slice_dataframe(
        all_data["day_ahead_price"], config.fit_start, config.end
    )
    params = get_split_params_from_config(sliced_data, config)

    return sliced_data, ExpandingWindowSplitter(**params)


def get_split_params_from_config(data: pd.DataFrame, config: BacktestingConfig) -> dict:
    """
    Generates the parameters for the ExpandingWindowSplitter based on the BacktestingConfig.
    These parameters determine the behaviour of the model predictive control (MPC) backtesting process,
    including the forecast horizon, initial training window, and step length.
    """

    if config.resolution == "hour":
        n_steps_per_hour = 1
    elif config.resolution == "quarterhour":
        n_steps_per_hour = 4
    else:
        raise ValueError(f"Unsupported resolution: {config.resolution}")

    # todo: make initial_window dynamic??

    initial_train = slice_dataframe(data, config.fit_start, config.opt_start)
    half_day = 12 * n_steps_per_hour
    initial_window = len(initial_train) + half_day
    return {
        "fh": range(
            1, config.forecast_horizon_days * 24 * n_steps_per_hour + half_day + 1
        ),
        "initial_window": initial_window,
        "step_length": config.step_length_days * 24 * n_steps_per_hour,
    }
