import pandas as pd
import pandera.pandas as pa


def get_datetime_index_schema_checks() -> list[pa.Check]:
    return [
        pa.Check(
            lambda s: isinstance(s.index, pd.DatetimeIndex),
            error="Index must be a DatetimeIndex",
        ),
        pa.Check(
            lambda s: s.index.is_monotonic_increasing,
            error="Index must be sorted in ascending order",
        ),
        pa.Check(
            lambda s: s.index.is_unique,
            error="Index must be unique",
        ),
        pa.Check(
            lambda s: not s.index.isnull().any(),
            error="Index values must not be null",
        ),
    ]
