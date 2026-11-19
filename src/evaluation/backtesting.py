from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from src.control.base import ControlConfig
from src.control.registry import get_controller_by_name
from src.data.raw import Resolution
from src.evaluation.forecasting import evaluate_forecast_accuracy
from src.evaluation.splitter import DateBasedExpandingWindowSplitter, offset_split
from src.forecasting.registry import get_forecaster_by_name
from src.evaluation.pnl import evaluate_pnl
from src.optimisation.registry import get_optimiser_by_name


@dataclass
class BacktestingConfig:
    fit_start: str
    opt_start: str
    end: str
    resolution: Resolution
    timezone: str = "Europe/Berlin"


def run_backtest(
    data: pd.DataFrame,
    forecaster_names: list[str],
    optimiser_name: str,
    controller_name: str,
    config: BacktestingConfig,
) -> pd.DataFrame:
    records = []
    states = {n: None for n in forecaster_names}

    controller = get_controller_by_name(controller_name)
    # todo: add option to pass controller kwargs
    splitter = init_splitter(config, controller.config)

    for train_idx, test_idx in tqdm(
        splitter.split(data),
        desc=f"Backtesting {optimiser_name} with forecasters {forecaster_names} using controller {controller_name}",
        leave=True,
    ):
        execution_date = data.index[test_idx[0]].tz_convert(config.timezone).normalize()
        train, test = offset_split(
            data, train_idx, test_idx, offset="12h"
        )  # todo: make offset configurable, move to config

        for name in forecaster_names:

            forecaster = get_forecaster_by_name(name, config.resolution)
            # todo: add option to pass forecaster kwargs
            optimiser = get_optimiser_by_name(
                optimiser_name, config.resolution, current_state=states[name]
            )  # todo: add option to pass optimiser kwargs
            forecasts, schedule, optimiser = controller.plan(
                forecaster, optimiser, train, test, execution_date
            )
            executed_schedule, states[name] = controller.execute(
                optimiser, schedule, execution_date
            )

            metric_values = evaluate_iteration(executed_schedule, forecasts, test)
            records.append(
                {"date": execution_date, "forecaster": name, **metric_values}
            )

        # todo: add option to save metrics to csv
        # todo: add option to persist forecasts and schedules to csv

    return pd.DataFrame(records)


def evaluate_iteration(
    executed_schedule: pd.DataFrame, forecasts: pd.Series, test: pd.DataFrame
) -> dict[str, float]:
    forecast_metrics = evaluate_forecast_accuracy(test, forecasts)
    pnl_metrics = evaluate_pnl(
        executed_schedule,
        test.reindex(executed_schedule.index),
        forecasts.reindex(executed_schedule.index),
    )
    return {**forecast_metrics, **pnl_metrics.sum().to_dict()}


def init_splitter(
    backtest_config: BacktestingConfig, control_config: ControlConfig
) -> DateBasedExpandingWindowSplitter:

    initial_train = pd.Timestamp(backtest_config.opt_start) - pd.Timestamp(
        backtest_config.fit_start
    )

    return DateBasedExpandingWindowSplitter(
        initial_train_days=initial_train.days,
        forecast_horizon_days=control_config.forecast_horizon_days,
        step_length_days=control_config.step_length_days,
        timezone=backtest_config.timezone,
    )
