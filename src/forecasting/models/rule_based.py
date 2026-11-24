import warnings
from abc import abstractmethod, ABC

import pandas as pd
from sktime.forecasting.naive import NaiveForecaster as NaiveForecasterSK

from src.data.raw import Resolution
from src.forecasting.models.base import ForecastingModel, ForecastVariables

warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.",
    category=UserWarning,
    module=r"sktime\.utils\.seasonality",
)


class PerfectForecaster(ForecastingModel):
    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)

    def fit(self, data: pd.DataFrame):
        pass

    def predict(self, data: pd.DataFrame):
        return data[self.variables.y]


class NaiveForecaster(ForecastingModel, ABC):
    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)
        self.estimator = NaiveForecasterWithoutRememberData(
            strategy="last", sp=self.lag
        )

    @property
    @abstractmethod
    def lag(self):
        raise NotImplementedError

    def fit(self, data: pd.DataFrame):
        freq = pd.infer_freq(data.index)
        self.estimator.fit(data[self.variables.y].asfreq(freq))

    def predict(self, data: pd.DataFrame) -> pd.Series:
        horizon = len(data)
        return self.estimator.predict(fh=range(1, horizon + 1))


class DailyNaiveForecaster(NaiveForecaster):
    @property
    def lag(self):
        return self.resolution.steps_per_day


class WeeklyNaiveForecaster(NaiveForecaster):
    @property
    def lag(self):
        return self.resolution.steps_per_week


class NaiveForecasterWithoutRememberData(NaiveForecasterSK):
    """Hacky subclass to suppress future warnings about the `remember_data`"""

    _config = {**NaiveForecasterSK._config, "remember_data": False}
