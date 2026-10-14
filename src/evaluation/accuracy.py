from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_forecast_accuracy(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {"MAE": mae, "RMSE": rmse}
