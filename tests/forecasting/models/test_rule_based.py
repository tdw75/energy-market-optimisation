import numpy as np
import pandas as pd

from src.data.raw import Resolution
from src.forecasting.models.base import ForecastVariables
from src.forecasting.models.rule_based import NaiveForecaster


class TestNaiveForecaster:
    def test_fit_predict(self):

        n = 24
        train_size = 12
        data = pd.DataFrame(
            {"value": range(0, n)}, index=pd.date_range("2023-01-01", periods=n)
        )
        data["value"] *= 10
        train = data.iloc[:train_size]
        test = data.iloc[train_size:]

        class MockNaiveForecaster(NaiveForecaster):
            @property
            def lag(self):
                return 6

        forecaster = MockNaiveForecaster(
            name="naive",
            variables=ForecastVariables(y="value"),
            resolution=Resolution.HOURLY,
        )
        forecaster.fit(train)

        pred = forecaster.predict(test)
        values_exp = np.concatenate(
            [
                train["value"].values[-forecaster.lag :],
                train["value"].values[-forecaster.lag :],
            ]
        )

        pred_exp = pd.Series(values_exp, index=test.index, dtype=float, name="value")
        pd.testing.assert_series_equal(pred, pred_exp)
