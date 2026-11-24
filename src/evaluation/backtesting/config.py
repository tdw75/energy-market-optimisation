import uuid
from dataclasses import dataclass
from datetime import datetime

from src.data.raw import Resolution


@dataclass
class BacktestingConfig:
    backtest_id: str
    run_idx: int
    fit_start: str
    opt_start: str
    end: str
    resolution: Resolution
    forecaster_name: str
    optimiser_name: str
    controller_name: str
    timezone: str = "Europe/Berlin"

    def __post_init__(self):
        self.run_id = self.generate_run_id(self.backtest_id)

    def generate_run_id(self, backtest_id: str) -> str:
        return f"{backtest_id}-{self.run_idx}-{self.forecaster_name}-{self.optimiser_name}-{self.controller_name}"

    def to_dict(self):
        dict_ = self.__dict__.copy()
        dict_["resolution"] = self.resolution.value
        return dict_


def generate_backtest_id(backtest_name: str) -> str:
    # only take the first 12 characters of the UUID to keep it short
    run_uuid = uuid.uuid4().hex[:12]
    today = datetime.now().strftime("%Y%m%d")
    return f"{backtest_name}-{today}-{run_uuid}"
