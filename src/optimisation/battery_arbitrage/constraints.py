from ortools.linear_solver.pywraplp import Solver

from src.optimisation.battery_arbitrage.config import Request
from src.optimisation.battery_arbitrage.variables import Variables


def define_constraints(
    solver: Solver, variables: Variables, request: Request
) -> Solver:

    solver = def_constr_state_of_charge(solver, variables, request)
    solver = def_constr_max_battery_capacity(solver, variables, request)
    solver = def_constr_max_charging_power(solver, variables, request)

    return solver


def def_constr_state_of_charge(solver: Solver, variables: Variables, request: Request):
    """
    Defines the state of charge at current time for each device as the sum of
    device's previous state of charge and current charging decision
    """

    for idx, t in enumerate(request.timesteps):

        if idx == 0:
            constr = solver.Constraint(
                request.current_state, request.current_state, f"state_of_charge-{t}"
            )
        else:
            t_prev = request.timesteps[idx - 1]
            constr = solver.Constraint(0, 0, f"state_of_charge-{t}")
            constr.SetCoefficient(variables.state[t_prev], -1)
        constr.SetCoefficient(variables.state[t], 1)
        constr.SetCoefficient(variables.charge[t], -1)
        constr.SetCoefficient(variables.discharge[t], 1)

    return solver


def def_constr_max_battery_capacity(
    solver: Solver, variables: Variables, request: Request
):
    """
    Defines the constraint that the state of charge at each time step cannot exceed the battery capacity
    """
    for t in request.timesteps:
        constr = solver.Constraint(
            -solver.infinity(), request.battery_capacity, f"max_battery_capacity-{t}"
        )
        constr.SetCoefficient(variables.state[t], 1)

    return solver


def def_constr_max_charging_power(
    solver: Solver, variables: Variables, request: Request
):
    """
    Defines the constraint that the charging and discharging decisions at each time step cannot exceed the maximum charging power
    """
    for t in request.timesteps:
        constr = solver.Constraint(
            -solver.infinity(), request.max_power, f"max_charging_power-{t}"
        )
        constr.SetCoefficient(variables.charge[t], 1)
        constr.SetCoefficient(variables.discharge[t], 1)

    return solver
