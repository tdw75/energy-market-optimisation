import pandas as pd
import pytest

from src.data.raw import Resolution
from src.optimisation.battery_arbitrage.config import BatteryConfig
from src.optimisation.battery_arbitrage.lp import BatteryArbitrageLP


class TestBatteryArbitrageLP:

    config = BatteryConfig(
        battery_capacity=10.0,
        max_power=5.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
    )

    @pytest.mark.parametrize(
        "resolution, value",
        [(Resolution.QUARTER_HOURLY, 1.25), (Resolution.HOURLY, 5.0)],
    )
    def test_trivial_lp(self, resolution, value):

        price = -100.0
        forecast = pd.Series([price], index=pd.DatetimeIndex(["2025-01-01 00:00:00"]))
        optimiser = BatteryArbitrageLP(
            config=self.config, current_state=0.0, resolution=resolution
        )
        schedule = optimiser.solve(forecast)

        assert schedule["buy"].iloc[0] == pytest.approx(value)
        assert schedule["sell"].iloc[0] == pytest.approx(0.0)
        assert schedule["state"].iloc[0] == pytest.approx(value)
        assert optimiser.solver.Objective().Value() == pytest.approx(-price * value)

    @pytest.mark.parametrize("state_exp, efficiency", [(5.0, 1.0), (4.5, 0.9)])
    def test_charging_efficiency_effect_on_state_constraint(self, state_exp, efficiency):

        self.config.charging_efficiency = efficiency

        price = -100.0
        forecast = pd.Series([price], index=pd.DatetimeIndex(["2025-01-01 00:00:00"]))
        optimiser = BatteryArbitrageLP(
            config=self.config, current_state=0.0, resolution=Resolution.HOURLY
        )
        schedule = optimiser.solve(forecast)

        assert schedule["buy"].iloc[0] == pytest.approx(5.0)
        assert schedule["sell"].iloc[0] == pytest.approx(0.0)
        assert schedule["state"].iloc[0] == pytest.approx(state_exp)
        assert optimiser.solver.Objective().Value() == pytest.approx(-price * 5.0)

    @pytest.mark.parametrize(
        "value, efficiency", [(1.25, 1.0), (1.125, 0.9)]
    )
    def test_discharging_efficiency_effect_on_obj_val(self, value, efficiency):

        self.config.discharging_efficiency = efficiency
        price = 100.0
        forecast = pd.Series([price], index=pd.DatetimeIndex(["2025-01-01 00:00:00"]))
        optimiser = BatteryArbitrageLP(
            config=self.config, current_state=5.0, resolution=Resolution.QUARTER_HOURLY
        )
        schedule = optimiser.solve(forecast)

        assert schedule["buy"].iloc[0] == pytest.approx(0.0)
        assert schedule["sell"].iloc[0] == pytest.approx(1.25)
        assert schedule["state"].iloc[0] == pytest.approx(5.0 - 1.25)
        assert optimiser.solver.Objective().Value() == pytest.approx(price * value)
