from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import DLinear, NLinear

from src.data.raw import Resolution
from src.forecasting.models.base import ForecastingModel, ForecastVariables


class NeuralForecaster(ForecastingModel, ABC):
    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)
        # self.horizon = (
        #     self.resolution.steps_per_week + self.resolution.steps_per_hour * 12
        # )
        self.input_size = self.resolution.steps_per_week
        self.nf: NeuralForecast = None

    @abstractmethod
    def _build_estimator(self, horizon: int):
        raise NotImplementedError

    def fit(self, data: pd.DataFrame):
        df = self.to_neuralforecast_df(data[self.variables.y])
        self.nf.fit(df)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return self.to_standard_output(self.nf.predict())

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame = None) -> pd.Series:
        estimator = self._build_estimator(horizon=len(test))
        self.nf = NeuralForecast(models=[estimator], freq=self.resolution.freq)
        return super().fit_predict(train, test)

    def to_neuralforecast_df(self, data: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {"unique_id": self.variables.y, "ds": data.index, "y": data.to_numpy()}
        )

    @property
    @abstractmethod
    def estimator_name(self) -> str:
        raise NotImplementedError

    def to_standard_output(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(
            data[self.estimator_name].to_numpy(dtype=np.float64),
            index=data["ds"],
            name=self.variables.y,
        )


class DLinearForecaster(NeuralForecaster):
    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)

    def _build_estimator(self, horizon: int) -> DLinear:
        return DLinear(
            h=horizon,
            input_size=self.input_size,
            loss=MAE(),
            scaler_type="robust",
            learning_rate=1e-3,
            max_steps=100,
            step_size=24,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )

    @property
    def estimator_name(self) -> str:
        return "DLinear"


class NLinearForecaster(NeuralForecaster):
    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)

    def _build_estimator(self, horizon: int) -> NLinear:
        return NLinear(
            h=horizon,
            input_size=self.input_size,
            loss=MAE(),
            scaler_type="robust",
            learning_rate=1e-3,
            max_steps=100,
            step_size=24,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )

    @property
    def estimator_name(self) -> str:
        return "NLinear"
