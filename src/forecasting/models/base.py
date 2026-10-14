from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from sktime.forecasting.naive import NaiveForecaster as NaiveForecasterSK

# todo: add better way to store and pass params than dictionary, e.g. dataclass or pydantic model


@dataclass
class ForecastVariables:
    y: str
    X: list[str] | None = None

    def __post_init__(self):
        if self.X is None:
            self.X = []


class ForecastingModel(ABC):
    def __init__(self, name: str, variables: ForecastVariables, params: dict = {}):
        self.name = name
        self.variables = variables
        self.params = params

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("Train method not implemented.")

    def predict(self, data: pd.DataFrame):
        raise NotImplementedError("Predict method not implemented.")

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame = None):
        if test is None:
            test = train
        self.fit(train)
        return self.predict(test)


class NaiveForecaster(ForecastingModel):
    def __init__(self, name: str, variables: ForecastVariables, params: dict = {}):
        super().__init__(name, variables, params=params)
        self.lag = self.params.get("lag", 1)
        self.estimator = NaiveForecasterSK(strategy="last", sp=self.lag)

    def fit(self, data: pd.DataFrame):
        self.estimator.fit(data[self.variables.y])

    def predict(self, data: pd.DataFrame):
        horizon = len(data)
        return self.estimator.predict(fh=range(1, horizon + 1))
