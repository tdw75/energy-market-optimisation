import pandas as pd
from tqdm import tqdm

from src.control.base import ControlConfig
from src.control.registry import get_controller_by_name
from src.evaluation.backtesting.config import BacktestingConfig
from src.evaluation.backtesting.results import BacktestResults
from src.evaluation.backtesting.splitter import (
    DateBasedExpandingWindowSplitter,
    offset_split,
)
from src.forecasting.registry import get_forecaster_by_name
from src.optimisation.battery_arbitrage.config import BatteryConfig
from src.optimisation.registry import get_optimiser_by_name


def run_backtest(
    data: pd.DataFrame, battery: BatteryConfig, config: BacktestingConfig
) -> BacktestResults:
    state = None
    results = BacktestResults(
        config=config,
        battery=battery,  # todo: add battery config to backtesting config
        price=pd.DataFrame(),
        schedule=pd.DataFrame(),
        state=pd.Series(),
        metrics=pd.DataFrame(),
    )

    controller = get_controller_by_name(config.controller_name)
    # todo: add option to pass controller kwargs
    splitter = init_splitter(config, controller.config)

    for train_idx, test_idx in tqdm(
        splitter.split(data),
        desc=f"Backtesting {config.optimiser_name} with forecaster {config.forecaster_name} using controller {config.controller_name}",
        leave=True,
    ):
        execution_date = data.index[test_idx[0]].tz_convert(config.timezone).normalize()
        train, test = offset_split(
            data, train_idx, test_idx, offset="12h"
        )  # todo: make offset configurable, move to config

        forecaster = get_forecaster_by_name(config.forecaster_name, config.resolution)
        # todo: add option to pass forecaster kwargs
        optimiser = get_optimiser_by_name(
            config.optimiser_name,
            config.resolution,
            current_state=state,
            **battery.__dict__,
        )  # todo: add option to pass optimiser kwargs
        forecasts, schedule, optimiser = controller.plan(
            forecaster, optimiser, train, test, execution_date
        )
        executed_schedule, state = controller.execute(
            optimiser, schedule, execution_date
        )

        results.store_prices(forecasts, test)
        results.store_schedule(schedule)
        results.store_state(state, execution_date)
        results.evaluate_iteration(schedule, forecasts, test, execution_date)

    return results


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
