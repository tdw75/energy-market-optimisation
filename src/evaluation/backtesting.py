import pandas as pd
from sktime.split.base import BaseWindowSplitter
from tqdm import tqdm

from src.evaluation.accuracy import evaluate_forecast_accuracy
from src.forecasting.models.base import ForecastingModel


def run_backtest(
    data: pd.DataFrame, forecaster: ForecastingModel, splitter: BaseWindowSplitter
) -> dict:
    # todo: only forecast for now, add optimisation later
    metrics = {}

    for split in tqdm(
        splitter.split(data), desc=f"Backtesting {forecaster.name}", leave=True
    ):
        # todo: make forecast start at 12pm

        train = data.iloc[split[0]]
        test = data.iloc[split[1]]
        date = test.index[-1].date()

        forecast = forecaster.fit_predict(train, test)
        metrics_split = evaluate_forecast_accuracy(forecast, test)
        metrics[date] = metrics_split
        # todo: add option to persist metrics/forecasts

    return metrics
