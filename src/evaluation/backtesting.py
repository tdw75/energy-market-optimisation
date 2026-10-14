import pandas as pd
from sktime.split.base import BaseWindowSplitter
from tqdm import tqdm

from src.evaluation.accuracy import evaluate_forecast_accuracy
from src.forecasting.models.base import ForecastingModel
from src.optimisation.battery_arbitrage.config import Config
from src.optimisation.battery_arbitrage.evaluation import evaluate_pnl
from src.optimisation.battery_arbitrage.lp import optimise, construct_request


def run_backtest(
    data: pd.DataFrame, forecaster: ForecastingModel, splitter: BaseWindowSplitter
) -> dict:
    metrics = {}
    current_state = 0.0
    config = Config(battery_capacity=10.0, max_power=1.0)

    for split in tqdm(
        splitter.split(data), desc=f"Backtesting {forecaster.name}", leave=True
    ):

        train = data.iloc[split[0]]
        test = data.iloc[split[1]]
        date = test.index[-1].date()

        # todo: need to make more general for different forecasters,
        #  e.g. some may require exogenous variables, some may not
        # FORECAST
        forecast = forecaster.fit_predict(train, test)  # forecast from 12pm
        metrics[date] = evaluate_forecast_accuracy(forecast, test)

        # todo: need to make more general for different strategies/applications
        # OPTIMISE
        # truncate forecast to only include the next day (from 00:00 to 23:59)
        forecast_opt = forecast[forecast.index.date == date]
        test_opt = test[test.index.date == date]
        request = construct_request(forecast_opt, current_state, config)
        schedule = optimise(request)
        current_state = schedule["state"].iloc[-1]
        pnl = evaluate_pnl(schedule, test_opt, forecast_opt)
        metrics[date].update(pnl.sum().to_dict())

    return pd.DataFrame.from_dict(metrics, orient="index")
