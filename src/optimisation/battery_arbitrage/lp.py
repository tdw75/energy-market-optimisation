import pandas as pd
from ortools.linear_solver import pywraplp

from src.optimisation.battery_arbitrage.config import Request, Config
from src.optimisation.battery_arbitrage.constraints import define_constraints
from src.optimisation.battery_arbitrage.variables import Variables, init_variables


def optimise(request: Request) -> pd.DataFrame:

    solver = pywraplp.Solver.CreateSolver("GLOP")
    solver, variables = init_variables(solver, request.timesteps)

    solver = define_objective_function(solver, variables, request)
    solver = define_constraints(solver, variables, request)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print(f"Objective value = {solver.Objective().Value():0.4f}")
    else:
        raise ValueError("The problem does not have an optimal solution.")

    schedule = get_schedule_from_solution(variables, request)

    return schedule


def define_objective_function(solver, variables: Variables, request: Request):
    """
    Defines the objective function for the battery arbitrage linear program.
    The objective is to maximise the profit from charging and discharging the battery based on the given prices.
    """
    objective = solver.Objective()
    for t in request.timesteps:
        objective.SetCoefficient(variables.charge[t], -request.prices[t])
        objective.SetCoefficient(variables.discharge[t], request.prices[t])
    objective.SetMaximization()
    return solver


def get_schedule_from_solution(variables: Variables, request: Request) -> pd.DataFrame:
    schedule = pd.DataFrame(index=request.timesteps)
    schedule["buy"] = [variables.charge[t].solution_value() for t in request.timesteps]
    schedule["sell"] = [
        variables.discharge[t].solution_value() for t in request.timesteps
    ]
    schedule["state"] = [variables.state[t].solution_value() for t in request.timesteps]
    return schedule


def construct_request(
    forecast: pd.Series, current_state: float, config: Config
) -> Request:
    request = {
        "timesteps": list(forecast.index),
        "prices": forecast.to_dict(),  # todo: make sure this is the right format for prices
        "battery_capacity": config.battery_capacity,
        "max_power": config.max_power,
        "current_state": current_state,
    }

    return Request(**request)
