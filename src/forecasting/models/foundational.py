from functools import lru_cache

import numpy as np
import pandas as pd
from timestral import TimestralForecaster as TimestralModel

from src.data.raw import Resolution
from src.forecasting.models.base import ForecastingModel, ForecastVariables


@lru_cache(maxsize=2)
def load_timestral(model_id: str, device: str):
    return TimestralModel.from_pretrained(model_id, device=device)


class Timestral5MForecaster(ForecastingModel):

    model_id: str = "mistralai/Timestral-5M"

    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params)
        device = self.params.get("device", "cpu")

        self.estimator = load_timestral(self.model_id, device)
        self.context = None

    def fit(self, data: pd.DataFrame):
        self.context = data[self.variables.y].to_numpy(dtype=np.float32)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.context is None:
            raise RuntimeError("Forecaster has not been fitted.")

        forecast = self.estimator.forecast(
            self.context,
            prediction_length=len(data),
        )

        median = forecast.median[0, 0]

        if hasattr(median, "detach"):
            median = median.detach().cpu().numpy()

        return pd.Series(
            np.asarray(median, dtype=np.float64),
            index=data.index,
            name=self.variables.y,
        )


class Timestral28MForecaster(Timestral5MForecaster):

    model_id: str = "mistralai/Timestral-28M"
