import pandas as pd
from ortools.linear_solver.pywraplp import Solver

from src.optimisation.battery_arbitrage.config import BatteryArbitrageRequest
from src.optimisation.battery_arbitrage.variables import Variables


def def_constr_state_of_charge(
    solver: Solver, variables: Variables, request: BatteryArbitrageRequest
):
    """
    Defines the state of charge at current time for each device as the sum of
    device's previous state of charge and current charging decision
    """

    for idx, t in enumerate(request.timesteps):

        constr = solver.Constraint(
            request.current_state if idx == 0 else 0,
            request.current_state if idx == 0 else 0,
            get_constr_name("state_of_charge", t),
        )
        constr.SetCoefficient(variables.state[t], 1)
        constr.SetCoefficient(variables.charge[t], -1 * request.charging_efficiency)
        constr.SetCoefficient(variables.discharge[t], 1)

        if idx > 0:
            t_prev = request.timesteps[idx - 1]
            constr.SetCoefficient(variables.state[t_prev], -1)

    return solver


def def_constr_max_battery_capacity(
    solver: Solver, variables: Variables, request: BatteryArbitrageRequest
):
    """
    Defines the constraint that the state of charge at each time step cannot exceed the battery capacity
    """
    for t in request.timesteps:
        constr = solver.Constraint(
            -solver.infinity(),
            request.battery_capacity,
            get_constr_name("max_battery_capacity", t),
        )
        constr.SetCoefficient(variables.state[t], 1)

    return solver


def def_constr_max_charging_power(
    solver: Solver,
    variables: Variables,
    request: BatteryArbitrageRequest,
    delta_t: float,
):
    """
    Defines the constraint that the charging and discharging decisions at each time step cannot exceed the maximum charging power
    """
    for t in request.timesteps:
        constr = solver.Constraint(
            -solver.infinity(),
            request.max_power * delta_t,
            get_constr_name("max_charging_power", t),
        )
        constr.SetCoefficient(variables.charge[t], 1)
        constr.SetCoefficient(variables.discharge[t], 1)

    return solver


def def_constr_terminal_state_of_charge(
    solver: Solver, variables: Variables, request: BatteryArbitrageRequest
):
    """
    Defines the constraint that the terminal state of charge at the last time step must be
    equal to the initial state of charge
    """
    T = request.timesteps[-1]
    constr = solver.Constraint(
        request.current_state,
        request.current_state,
        get_constr_name("terminal_state_of_charge", T),
    )
    constr.SetCoefficient(variables.state[T], 1)
    # todo: investigate other approaches for the terminal state, e.g. some sort of present value of future prices

    return solver


def get_constr_name(name: str, t: pd.Timestamp) -> str:
    return f"{name}-{t.strftime('%Y-%m-%d %X')}"
