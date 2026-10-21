from unittest.mock import patch

import pandas as pd
import pytest

from src.evaluation.backtesting import BacktestingConfig, get_split_params_from_config


@pytest.mark.parametrize(
    ("config", "training_rows", "expected"),
    [
        (
            BacktestingConfig(
                fit_start="2023-01-01",
                opt_start="2023-01-08",
                end="2023-02-01",
                resolution="hour",
            ),
            24 * 7,  # 7 days of hourly data
            {
                "fh": range(1, 24 * 7 + 12 + 1),  # 7 days + half day
                "initial_window": 24 * 7 + 12,  # 7 days + half day
                "step_length": 24 * 1,  # 1 day of hourly data
            },
        ),
        (
            BacktestingConfig(
                fit_start="2023-01-01",
                opt_start="2023-01-08",
                end="2023-02-01",
                resolution="quarterhour",
            ),
            24 * 7 * 4,  # 7 days of quarter-hourly data
            {
                "fh": range(1, 24 * 7 * 4 + 12 * 4 + 1),  # 7 days + half day
                "initial_window": 24 * 7 * 4 + 12 * 4,  # 7 days + half day
                "step_length": 24 * 4,
            },
        ),
        (
            BacktestingConfig(
                fit_start="2023-01-01",
                opt_start="2023-01-15",
                end="2023-03-01",
                resolution="hour",
                step_length_days=2,
                forecast_horizon_days=3,
            ),
            14 * 24,  # 14 days of hourly data
            {
                "fh": range(1, 3 * 24 + 12 + 1),  # 3 days + 12 hours
                "initial_window": 348,  # initial window + half day
                "step_length": 48,  # 2 days * 24 hours
            },
        ),
    ],
)
def test_get_split_params_from_config(config, training_rows, expected):
    data = pd.DataFrame(index=pd.date_range("2023-01-01", periods=training_rows))
    sliced_training_data = pd.DataFrame(index=range(training_rows))

    with patch(
        "src.evaluation.backtesting.slice_dataframe",
        return_value=sliced_training_data,
    ) as slice_dataframe:
        params = get_split_params_from_config(data, config)

    assert params == expected
    slice_dataframe.assert_called_once_with(data, config.fit_start, config.opt_start)
