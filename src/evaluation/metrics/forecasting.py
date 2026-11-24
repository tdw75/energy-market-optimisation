import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np


def evaluate_forecast_accuracy(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mhd": min_max_hour_deviation(y_true, y_pred),
        "mpd": min_max_price_deviation(y_true, y_pred),
    }


def min_max_hour_deviation(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> float:

    min_idx_true = np.argmin(y_true)
    max_idx_true = np.argmax(y_true)

    min_idx_pred = np.argmin(y_pred)
    max_idx_pred = np.argmax(y_pred)

    return (abs(min_idx_pred - min_idx_true) + abs(max_idx_pred - max_idx_true)) / 2


def min_max_price_deviation(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> float:

    min_true = np.min(y_true)
    max_true = np.max(y_true)

    min_pred = np.min(y_pred)
    max_pred = np.max(y_pred)

    return (abs(min_pred - min_true) + abs(max_pred - max_true)) / 2


# todo: add more metrics that better correlate with the actual performance of the strategy,
#  e.g. some notion of rank similarity
