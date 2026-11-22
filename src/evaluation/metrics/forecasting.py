from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np


def evaluate_forecast_accuracy(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mhd": min_max_hour_deviation(y_true, y_pred),
    }


def min_max_hour_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    min_idx_true = np.argmin(y_true)
    max_idx_true = np.argmax(y_true)

    min_idx_pred = np.argmin(y_pred)
    max_idx_pred = np.argmax(y_pred)

    return float(abs(min_idx_pred - min_idx_true) + abs(max_idx_pred - max_idx_true))



# todo: add more metrics that better correlate with the actual performance of the strategy,
#  e.g. some notion of rank similarity
