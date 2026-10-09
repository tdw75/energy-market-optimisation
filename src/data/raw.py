from collections import namedtuple

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DatasetFilter = namedtuple("DatasetFilter", ["filter", "name", "column", "description"])
dataset_filters = {
    "day_ahead_price": DatasetFilter(
        4169, "day_ahead_price", "price", "Day-ahead electricity prices"
    ),
    "total_load": DatasetFilter(
        410, "total_load", "total_load_mwh", "Total energy consumption"
    ),
    "residual_load": DatasetFilter(
        4359, "residual_load", "residual_load_mwh", "Residual load"
    ),
    "wind_onshore": DatasetFilter(
        4067, "wind_onshore", "onshore_wind_mwh", "Onshore wind energy generation"
    ),
    "wind_offshore": DatasetFilter(
        1225, "wind_offshore", "offshore_wind_mwh", "Offshore wind energy generation"
    ),
    "solar": DatasetFilter(4068, "solar", "solar_mwh", "Solar energy generation"),
    "gas": DatasetFilter(4071, "gas", "gas_mwh", "Gas energy generation"),
}


def get_data_from_smard(
    dataset: DatasetFilter,
    start_date: str,
    resolution: str,
    is_save: bool = True,
) -> pd.DataFrame:
    index_url = f"https://www.smard.de/app/chart_data/{dataset.filter}/DE/index_{resolution}.json"
    index_data = requests.get(index_url).json()

    start = (
        pd.Timestamp(f"{start_date} 00:00:00", tz="UTC") - pd.Timedelta(weeks=1)
    ).timestamp() * 1000

    df_list = []
    timestamps = [t for t in index_data["timestamps"] if t >= start]
    for ts in tqdm(timestamps, desc=f"Timestamp for {dataset.name}", leave=True):
        data_url = (
            f"https://www.smard.de/app/chart_data/{dataset.filter}/DE/"
            f"{dataset.filter}_DE_{resolution}_{ts}.json"
        )

        data = requests.get(data_url).json()
        df_ts = pd.DataFrame(
            data["series"],
            columns=[
                "timestamp",
                dataset.column,
            ],  # todo: make this dynamic based on dataset.name
        )
        df_list.append(df_ts)

    df = pd.concat(df_list, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.loc[df["timestamp"] >= start_date]

    if is_save:
        root = Path(__file__).resolve().parent.parent
        data_dir = root / "data" / "raw" / resolution
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / f"{dataset.name}.csv", index=False)

    return df


def read_raw_data(names: list[str], resolution: str) -> dict[str, pd.DataFrame]:
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data" / "raw" / resolution
    data_dict = {}
    for name in names:
        data_dict[name] = pd.read_csv(
            data_dir / f"{name}.csv", index_col="timestamp", parse_dates=["timestamp"]
        )

    return data_dict


def slice_dataframe(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Slices the DataFrame from start_date (inclusive) to end_date (exclusive).
    """
    return df[(df.index >= start_date) & (df.index < end_date)]
