import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from sktime.forecasting.compose import make_reduction
from sktime.transformations.lag import Lag

from src.data.raw import Resolution
from src.forecasting.models.base import (
    ForecastingModel,
    ForecastVariables,
    SktimeForecaster,
)


class BoostedForecasterWithWindow(SktimeForecaster):

    def _build_estimator(self):
        params = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "random_seed": 42,
        }

        window_length = self.params.get(
            "window_length",
            self.resolution.steps_per_week * 1,
        )

        model_params = {k: v for k, v in self.params.items() if k != "window_length"}
        params.update(model_params)
        regressor = CatBoostRegressor(**params)

        return make_reduction(
            estimator=regressor,
            strategy="recursive",
            window_length=window_length,
            scitype="tabular-regressor",
        )


class BoostedForecasterWithLags(ForecastingModel):
    """
    Boosted forecaster that uses CatBoostRegressor with lag features. Included lag features are the entire past day (24 hours)
    and the seasonal lag timepoint of the past 7 days.
    """

    def __init__(
        self,
        name: str,
        variables: ForecastVariables,
        resolution: Resolution,
        params: dict = None,
    ):
        super().__init__(name, variables, resolution, params=params)
        self.estimator = self._build_estimator()

    def fit(self, data: pd.Series):
        y = data[self.variables.y].astype(np.float64)
        X = self._make_lag_features(y)
        self.estimator.fit(X, y)

    def predict(self, data: pd.Series, **kwargs) -> pd.Series:
        history: pd.Series = kwargs["history"].copy()

        for timestamp in data.index:
            next_ts = pd.Series([np.nan], index=[timestamp], name=self.variables.y)
            X = self._make_lag_features(pd.concat([history, next_ts])).loc[[timestamp]]
            history.loc[timestamp] = float(self.estimator.predict(X)[0])

        return history.loc[data.index]

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame = None) -> pd.Series:
        if test is None:
            test = train
        self.fit(train)
        preds = self.predict(test, history=train[self.variables.y])
        return self.validate_predictions(preds, test)

    @staticmethod
    def _build_estimator():
        return CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def _make_lag_features(self, y: pd.Series) -> pd.DataFrame:
        steps = self.resolution.steps_per_day
        past_day = [*range(1, steps)]
        seasonal_daily = [i * steps for i in range(1, 8)]
        transformer = Lag(lags=past_day + seasonal_daily, index_out="original")

        return transformer.fit_transform(y)
