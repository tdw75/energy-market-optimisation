from dataclasses import dataclass

from src.data.raw import Resolution


@dataclass
class BacktestingConfig:
    run_id: str
    fit_start: str
    opt_start: str
    end: str
    resolution: Resolution
    forecaster_name: str
    optimiser_name: str
    controller_name: str
    timezone: str = "Europe/Berlin"
