from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

# todo: add better way to store and pass params than dictionary, e.g. dataclass or pydantic model


@dataclass
class ForecastVariables:
    y: str
    X: list[str] | None = None

    def __post_init__(self):
        if self.X is None:
            self.X = []


class ForecastingModel(ABC):
    def __init__(self, name: str, variables: ForecastVariables, params: dict = None):
        self.name = name
        self.variables = variables
        self.params = params or {}

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("Train method not implemented.")

    @abstractmethod
    def predict(self, data: pd.DataFrame):
        raise NotImplementedError("Predict method not implemented.")

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame = None):
        if test is None:
            test = train
        self.fit(train)
        return self.predict(test)
