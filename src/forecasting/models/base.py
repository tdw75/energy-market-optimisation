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
    def predict(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Predict method not implemented.")

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame = None) -> pd.Series:
        if test is None:
            test = train
        self.fit(train)
        preds = self.predict(test)
        return self.validate_predictions(preds, test)

    def validate_predictions(self, preds: pd.Series, data: pd.DataFrame) -> pd.Series:

        if not isinstance(preds, pd.Series):
            raise TypeError("Predictions must be a pandas Series.")

        if not preds.index.equals(data.index):
            raise ValueError("Prediction index must match input index.")

        if preds.isna().any():
            raise ValueError("Predictions contain missing values.")

        preds.name = self.variables.y
        return preds
