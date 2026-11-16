from dataclasses import dataclass

import pandas as pd
from sktime.split import ExpandingWindowSplitter
from tqdm import tqdm

from src.control.base import ControlConfig
from src.control.registry import get_controller_by_name
from src.data.raw import slice_dataframe, Resolution
from src.evaluation.accuracy import evaluate_forecast_accuracy
from src.forecasting.registry import get_forecaster_by_name
from src.optimisation.battery_arbitrage.evaluation import evaluate_pnl
from src.optimisation.registry import get_optimiser_by_name


@dataclass
class BacktestingConfig:
    fit_start: str
    opt_start: str
    end: str
    resolution: Resolution


def run_backtest(
    raw_data: pd.DataFrame,
    forecaster_names: list[str],
    optimiser_name: str,
    controller_name: str,
    config: BacktestingConfig,
) -> pd.DataFrame:
    records = []
    states = {n: None for n in forecaster_names}

    controller = get_controller_by_name(controller_name)
    data, splitter = init_splitter(raw_data, config, controller.config)

    for split in tqdm(
        splitter.split(data),
        desc=f"Backtesting {optimiser_name} with forecasters {forecaster_names} using controller {controller_name}",
        leave=True,
    ):

        train = data.iloc[split[0]]
        test = data.iloc[split[1]]
        date = test.index[splitter.step_length + 1].date()

        for name in forecaster_names:

            forecaster = get_forecaster_by_name(name)
            optimiser = get_optimiser_by_name(
                optimiser_name, config.resolution, current_state=states[name]
            )
            forecasts, schedule, optimiser = controller.plan(
                forecaster, optimiser, train, test, date
            )
            executed_schedule, states[name] = controller.execute(optimiser, schedule, date)

            metric_values = evaluate_iteration(executed_schedule, forecasts, test, date)
            records.append({"date": date, "forecaster": name, **metric_values})

        # todo: add option to save metrics to csv
        # todo: add option to persist forecasts and schedules to csv

    return pd.DataFrame(records)


def evaluate_iteration(
    executed_schedule: pd.DataFrame,
    forecasts: pd.DataFrame,
    test: pd.DataFrame,
    date: pd.Timestamp,
):
    forecast_metrics = evaluate_forecast_accuracy(forecasts, test)
    pnl_metrics = evaluate_pnl(
        executed_schedule,
        # todo: align with executed schedule/control step length
        test[test.index.date == date],
        # todo: align with executed schedule/control step length
        forecasts[forecasts.index.date == date],
    )
    return {**forecast_metrics, **pnl_metrics.sum().to_dict()}


def init_splitter(
    all_data: pd.DataFrame,
    backtest_config: BacktestingConfig,
    control_config: ControlConfig,
) -> tuple[pd.DataFrame, ExpandingWindowSplitter]:

    # todo: this is actually part of the control logic, should maybe move to MPC module

    sliced_data = slice_dataframe(
        all_data, backtest_config.fit_start, backtest_config.end
    )
    params = get_split_params_from_config(sliced_data, backtest_config, control_config)

    return sliced_data, ExpandingWindowSplitter(**params)


def get_split_params_from_config(
    data: pd.DataFrame,
    backtest_config: BacktestingConfig,
    control_config: ControlConfig,
) -> dict:
    """
    Generates the parameters for the ExpandingWindowSplitter based on the BacktestingConfig.
    These parameters determine the behaviour of the model predictive control (MPC) backtesting process,
    including the forecast horizon, initial training window, and step length.
    """

    initial_train = slice_dataframe(
        data, backtest_config.fit_start, backtest_config.opt_start
    )
    half_day = 12 * backtest_config.resolution.steps_per_hour
    initial_window = len(initial_train) + half_day
    return {
        "fh": range(
            1,
            control_config.forecast_horizon_days * backtest_config.resolution.steps_per_day + half_day + 1,
        ),
        "initial_window": initial_window,
        "step_length": control_config.step_length_days * backtest_config.resolution.steps_per_day,
    }
