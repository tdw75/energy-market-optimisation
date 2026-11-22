import pandas as pd
from src.data.raw import Resolution
from src.optimisation.battery_arbitrage.config import BatteryConfig
from src.optimisation.battery_arbitrage.lp import BatteryArbitrageLP
from src.optimisation.battery_arbitrage.variables import Variables


def test_init_variables():
    ts_strings = ["2025-01-01 00:00:00", "2025-01-01 01:00:00", "2025-01-01 02:00:00"]
    forecast = pd.Series([100.0, 200.0, 300.0], index=pd.DatetimeIndex(ts_strings))
    config = BatteryConfig(
        battery_capacity=10.0,
        max_power=5.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
    )
    optimiser = BatteryArbitrageLP(
        config=config, current_state=0.0, resolution=Resolution.HOURLY
    )
    request = optimiser.construct_request(forecast)
    variables = optimiser.init_variables(request)

    var_names = []
    for var_type in ["state", "charge", "discharge"]:
        var_names.extend([f"{var_type}-{ts}" for ts in ts_strings])

    assert [v.forecaster_name() for v in optimiser.solver.variables()] == var_names
    assert [variables.state[t] for t in request.timesteps] == [
        optimiser.solver.variable(i) for i in range(3)
    ]

    assert [variables.charge[t] for t in request.timesteps] == [
        optimiser.solver.variable(i) for i in range(3, 6)
    ]

    assert [variables.discharge[t] for t in request.timesteps] == [
        optimiser.solver.variable(i) for i in range(6, 9)
    ]
    assert isinstance(variables, Variables)
