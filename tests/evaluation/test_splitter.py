import numpy as np
import pandas as pd
import pytest

from src.evaluation.splitter import (
    DateBasedExpandingWindowSplitter,
    clip_to_local_midnight,
)


class TestDateBasedExpandingWindowSplitter:
    @pytest.mark.parametrize(
        "start, train_days, expected_intervals",
        [
            ("2025-03-28", 2, 92),  # Spring DST, 23 hours
            ("2025-03-28", 1, 96),  # Normal day, 24 hours
            ("2025-10-24", 2, 100),  # Autumn DST, 25 hours
        ],
    )
    def test_calendar_day_length(self, start, train_days, expected_intervals):
        start = pd.Timestamp(start, tz="Europe/Berlin")
        end = start + pd.DateOffset(days=4) - pd.Timedelta(minutes=15)

        index = pd.date_range(start, end, freq="15min").tz_convert("UTC")
        data = pd.DataFrame({"price": 100}, index=index)

        splitter = DateBasedExpandingWindowSplitter(
            initial_train_days=train_days,
            forecast_horizon_days=1,
            expected_freq="15min",
        )

        _, test = next(splitter.split(data))

        assert len(test) == expected_intervals

    def test_training_window_expands(self):
        index = pd.date_range("2024-12-31", periods=96 * 7, freq="15min", tz="UTC")
        data = pd.DataFrame({"price": 100}, index=index)
        data = clip_to_local_midnight(data)

        splitter = DateBasedExpandingWindowSplitter(
            initial_train_days=2,
            forecast_horizon_days=2,
            step_length_days=1,
        )

        splits = list(splitter.split(data))
        train_lengths = [len(train) for train, _ in splits]

        assert all(a < b for a, b in zip(train_lengths, train_lengths[1:]))

    def test_missing_intervals_raise_error(self):
        index = pd.date_range("2024-12-31", periods=300, freq="15min", tz="UTC").delete(
            100
        )
        data = pd.DataFrame({"price": 100}, index=index)
        data = clip_to_local_midnight(data)

        splitter = DateBasedExpandingWindowSplitter(
            initial_train_days=1, forecast_horizon_days=1
        )

        with pytest.raises(ValueError, match="regular intervals"):
            list(splitter.split(data))
