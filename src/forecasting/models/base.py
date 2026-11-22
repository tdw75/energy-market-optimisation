from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import pandera as pa

from src.utils import get_datetime_index_schema_checks


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

        preds = self.schema.validate(preds)

        if not preds.index.equals(data.index):
            raise ValueError("Forecast index does not match input index.")
        return preds

    @property
    def schema(self) -> pa.SeriesSchema:
        return pa.SeriesSchema(
            float,
            nullable=False,
            name=self.variables.y,
            checks=get_datetime_index_schema_checks(),
        )
