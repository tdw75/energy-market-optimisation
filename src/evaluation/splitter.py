from numbers import Integral

import numpy as np
import pandas as pd
from sktime.split.base import BaseSplitter


class DateBasedExpandingWindowSplitter(BaseSplitter):
    def __init__(
        self,
        initial_train_days: int,
        forecast_horizon_days: int,
        step_length_days: int = 1,
        timezone: str = "Europe/Berlin",
        expected_freq: str | None = None,
    ):
        self.initial_train_days = initial_train_days
        self.forecast_horizon_days = forecast_horizon_days
        self.step_length_days = step_length_days
        self.timezone = timezone
        self.expected_freq = expected_freq
        super().__init__()

    def _validate_params(self):
        if (
            min(
                self.initial_train_days,
                self.forecast_horizon_days,
                self.step_length_days,
            )
            <= 0
        ):
            raise ValueError("Window lengths must be positive.")

        for name in (
            "initial_train_days",
            "forecast_horizon_days",
            "step_length_days",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    def _validate_index(self, y: pd.DatetimeIndex):
        if not isinstance(y, pd.DatetimeIndex) or y.tz is None:
            raise ValueError("Expected timezone-aware DatetimeIndex.")

        if not y.is_monotonic_increasing or not y.is_unique:
            raise ValueError("Index must be sorted and unique.")

        local_index = y.tz_convert(self.timezone)
        utc_index = y.tz_convert("UTC")

        if local_index[0] != local_index[0].normalize():
            raise ValueError("Index must start at local midnight.")

        diffs = np.diff(utc_index.as_unit("ns").asi8)
        if len(diffs) == 0 or not np.all(diffs == diffs[0]):
            raise ValueError("Index must have regular intervals.")

        freq = pd.Timedelta(diffs[0], unit="ns")
        if self.expected_freq and freq != pd.Timedelta(self.expected_freq):
            raise ValueError(f"Expected frequency {self.expected_freq}, got {freq}.")

        return local_index, utc_index, freq

    def _split(self, y: pd.DatetimeIndex):
        self._validate_params()
        local_index, utc_index, freq = self._validate_index(y)
        test_start = local_index[0] + pd.DateOffset(days=self.initial_train_days)

        while True:
            test_end = test_start + pd.DateOffset(days=self.forecast_horizon_days)
            start_utc, end_utc = test_start.tz_convert("UTC"), test_end.tz_convert(
                "UTC"
            )

            if end_utc - freq > utc_index[-1]:
                break

            train = np.flatnonzero(utc_index < start_utc)
            test = np.flatnonzero((utc_index >= start_utc) & (utc_index < end_utc))

            if not len(train) or not len(test):
                raise ValueError("Empty train or test window.")

            if utc_index[test[0]] != start_utc or utc_index[test[-1]] != end_utc - freq:
                raise ValueError("Incomplete test window.")

            yield train, test
            test_start += pd.DateOffset(days=self.step_length_days)

    def _fh(self):
        return None

    def get_cutoffs(self, y=None):
        if y is None:
            raise ValueError("y is required.")
        return np.array([train[-1] for train, _ in self.split(y)])


def clip_to_local_midnight(
    data: pd.Series | pd.DataFrame,
    timezone: str = "Europe/Berlin",
) -> pd.Series | pd.DataFrame:
    """Remove observations before the first complete local calendar day."""
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("Data must have a timezone-aware DatetimeIndex.")

    local = data.index.tz_convert(timezone)
    first_midnight = local[0].normalize()

    if local[0] == first_midnight:
        return data

    next_midnight = first_midnight + pd.DateOffset(days=1)
    return data.loc[data.index >= next_midnight.tz_convert("UTC")]
