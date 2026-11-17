from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from src.control.base import ControlConfig
from src.control.registry import get_controller_by_name
from src.data.raw import slice_dataframe, Resolution
from src.evaluation.accuracy import evaluate_forecast_accuracy
from src.evaluation.splitter import DateBasedExpandingWindowSplitter, offset_split
from src.forecasting.registry import get_forecaster_by_name
from src.optimisation.battery_arbitrage.evaluation import evaluate_pnl
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
    splitter = init_splitter(config, controller.config)

    for split in tqdm(
        splitter.split(data),
        desc=f"Backtesting {optimiser_name} with forecasters {forecaster_names} using controller {controller_name}",
        leave=True,
    ):
        train, test = offset_split(data, split[1], offset="12h")
        execution_date = (test.index[0] + pd.Timedelta(days=1)).date()

        for name in forecaster_names:

            forecaster = get_forecaster_by_name(name, config.resolution)
            optimiser = get_optimiser_by_name(
                optimiser_name, config.resolution, current_state=states[name]
            )
            forecasts, schedule, optimiser = controller.plan(
                forecaster, optimiser, train, test, execution_date
            )
            executed_schedule, states[name] = controller.execute(
                optimiser, schedule, execution_date
            )

            metric_values = evaluate_iteration(
                executed_schedule, forecasts, test, execution_date
            )
            records.append(
                {"date": execution_date, "forecaster": name, **metric_values}
            )

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
