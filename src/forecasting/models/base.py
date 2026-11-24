from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera as pa

from src.data.raw import Resolution
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

    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        self.name = name
        self.variables = variables
        self.resolution = resolution
        self.params = params or {}

    @abstractmethod
    def fit(self, data: pd.DataFrame | pd.Series):
        raise NotImplementedError("Train method not implemented.")

    @abstractmethod
    def predict(self, data: pd.DataFrame | pd.Series, **kwargs) -> pd.Series:
        raise NotImplementedError("Predict method not implemented.")

    def fit_predict(
        self, train: pd.DataFrame | pd.Series, test: pd.DataFrame | pd.Series = None
    ) -> pd.Series:
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


class SktimeForecaster(ForecastingModel, ABC):

    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)
        self.estimator = self._build_estimator()

    @abstractmethod
    def _build_estimator(self):
        raise NotImplementedError

    def fit(self, data: pd.DataFrame):
        self.estimator.fit(data[self.variables.y])

    def predict(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        fh = np.arange(1, len(data) + 1)
        preds = self.estimator.predict(fh=fh)

        return pd.Series(
            np.asarray(preds).reshape(-1).astype(np.float64),
            # todo: check if to_numpy is needed here, or if it can be removed
            index=data.index,
            name=self.variables.y,
        )
