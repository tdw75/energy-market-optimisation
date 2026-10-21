import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Solver

from src.optimisation.base import BaseORToolsLP
from src.optimisation.battery_arbitrage.config import (
    BatteryArbitrageRequest,
    BatteryArbitrageConfig,
)
from src.optimisation.battery_arbitrage.constraints import (
    def_constr_state_of_charge,
    def_constr_max_battery_capacity,
    def_constr_max_charging_power,
)
from src.optimisation.battery_arbitrage.variables import Variables, init_variables


class BatteryArbitrageLP(BaseORToolsLP):
    def __init__(self, config: BatteryArbitrageConfig, current_state: float):
        super().__init__(config, current_state)

    def solve(self, forecast: pd.Series) -> pd.DataFrame:

        solver = pywraplp.Solver.CreateSolver("GLOP")
        request = self.construct_request(forecast)

        solver, variables = self.init_variables(solver, request)
        solver = self.define_objective_function(solver, variables, request)
        solver = self.define_constraints(solver, variables, request)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise ValueError("The problem does not have an optimal solution.")

        schedule = self.get_schedule_from_solution(variables, request)
        return schedule

    def define_objective_function(
        self, solver: Solver, variables: Variables, request: BatteryArbitrageRequest
    ) -> Solver:
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

    def init_variables(
        self, solver: Solver, request: BatteryArbitrageRequest
    ) -> tuple[Solver, Variables]:
        solver, variables = init_variables(solver, request.timesteps)
        return solver, variables

    def define_constraints(
        self, solver, variables: Variables, request: BatteryArbitrageRequest
    ) -> Solver:
        solver = def_constr_state_of_charge(solver, variables, request)
        solver = def_constr_max_battery_capacity(solver, variables, request)
        solver = def_constr_max_charging_power(solver, variables, request)
        return solver

    def get_schedule_from_solution(
        self, variables: Variables, request: BatteryArbitrageRequest
    ) -> pd.DataFrame:
        ts = request.timesteps
        schedule = pd.DataFrame(index=ts)
        schedule["buy"] = [variables.charge[t].solution_value() for t in ts]
        schedule["sell"] = [variables.discharge[t].solution_value() for t in ts]
        schedule["state"] = [variables.state[t].solution_value() for t in ts]
        return schedule

    def construct_request(self, forecast: pd.Series) -> BatteryArbitrageRequest:
        request = {
            "timesteps": list(forecast.index),
            "prices": forecast.to_dict(),  # todo: make sure this is the right format for prices
            "battery_capacity": self.config.battery_capacity,
            "max_power": self.config.max_power,
            "current_state": self.current_state,
        }

        return BatteryArbitrageRequest(**request)

    def update_state(self, schedule: pd.DataFrame) -> pd.DataFrame:
        return schedule["state"].iloc[-1]
