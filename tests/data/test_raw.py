import pytest

from src.data.raw import Resolution


@pytest.mark.parametrize(
    "resolution, expected_minutes, expected_hours, expected_steps_per_hour, expected_steps_per_day, expected_steps_per_week",
    [
        (Resolution.HOURLY, 60, 1.0, 1, 24, 168),
        (Resolution.QUARTER_HOURLY, 15, 0.25, 4, 96, 672),
    ],
)
def test_resolution_properties(
    resolution,
    expected_minutes,
    expected_hours,
    expected_steps_per_hour,
    expected_steps_per_day,
    expected_steps_per_week,
):
    assert resolution.minutes == expected_minutes
    assert resolution.hours == expected_hours
    assert resolution.steps_per_hour == expected_steps_per_hour
    assert resolution.steps_per_day == expected_steps_per_day
    assert resolution.steps_per_week == expected_steps_per_week


def test_resolution_members():
    assert list(Resolution) == [
        Resolution.HOURLY,
        Resolution.QUARTER_HOURLY,
    ]


def test_resolution_values():
    assert Resolution.HOURLY.value == "hour"
    assert Resolution.QUARTER_HOURLY.value == "quarterhour"
