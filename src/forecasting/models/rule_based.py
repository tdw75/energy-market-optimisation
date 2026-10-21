import warnings

import pandas as pd
from sktime.forecasting.naive import NaiveForecaster as NaiveForecasterSK

from src.forecasting.models.base import ForecastingModel, ForecastVariables

warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.",
    category=UserWarning,
    module=r"sktime\.utils\.seasonality",
)


class PerfectForecaster(ForecastingModel):
    def __init__(self, name: str, variables: ForecastVariables, params: dict = None):
        super().__init__(name, variables, params=params)

    def fit(self, data: pd.DataFrame):
        pass

    def predict(self, data: pd.DataFrame):
        return data[self.variables.y]


class NaiveForecaster(ForecastingModel):
    def __init__(self, name: str, variables: ForecastVariables, params: dict = None):
        super().__init__(name, variables, params=params)
        self.lag = self.params.get("sp", 168)
        self.estimator = NaiveForecasterWithoutRememberData(
            strategy="last", sp=self.lag
        )

    def fit(self, data: pd.DataFrame):
        freq = pd.infer_freq(data.index)
        self.estimator.fit(data[self.variables.y].asfreq(freq))

    def predict(self, data: pd.DataFrame):
        horizon = len(data)
        return self.estimator.predict(fh=range(1, horizon + 1))


class NaiveForecasterWithoutRememberData(NaiveForecasterSK):
    """Hacky subclass to suppress future warnings about the `remember_data`"""
    _config = {**NaiveForecasterSK._config, "remember_data": False}
