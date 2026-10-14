import numpy as np
import pandas as pd

from src.forecasting.models.base import ForecastVariables
from src.forecasting.models.rule_based import NaiveForecaster


class TestNaiveForecaster:
    def test_fit_predict(self):
        n = 24
        sp = 6
        train_size = 12
        data = pd.DataFrame(
            {"value": range(0, n)}, index=pd.date_range("2023-01-01", periods=n)
        )
        data["value"] *= 10
        train = data.iloc[:train_size]
        test = data.iloc[train_size:]

        forecaster = NaiveForecaster(
            name="naive", variables=ForecastVariables(y="value"), params={"sp": sp}
        )
        forecaster.fit(train)

        pred = forecaster.predict(test)
        values_exp = np.concatenate(
            [train["value"].values[-sp:], train["value"].values[-sp:]]
        )

        pred_exp = pd.Series(values_exp, index=test.index, dtype=float, name="value")
        pd.testing.assert_series_equal(pred, pred_exp)
