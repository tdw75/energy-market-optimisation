from sktime.forecasting.sarimax import SARIMAX

from src.forecasting.models.base import SktimeForecaster


class SARIMAForecaster(SktimeForecaster):
    # fixme: super slow to train such that backtesting runs take too long

    def _build_estimator(self) -> SARIMAX:
        # todo: add auto arima or some other way to find best params for SARIMA model

        params = {
            "order": (1, 0, 0),
            "seasonal_order": (1, 0, 0, self.resolution.steps_per_day),
            "trend": "c",
            "maxiter": 20,
            "low_memory": True,
            "method": "lbfgs",
            "disp": False,
            "freq": self.resolution.freq,
        }
        params.update(self.params)

        return SARIMAX(**params)
