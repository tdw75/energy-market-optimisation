from abc import ABC, abstractmethod
from typing import Any

from ortools.linear_solver.pywraplp import Solver

import pandas as pd


class BaseORToolsLP(ABC):
    def __init__(self, config, current_state: Any):
        self.config = config
        self.current_state = current_state

    @abstractmethod
    def solve(self, forecast: pd.Series) -> Any:
        raise NotImplementedError

    @abstractmethod
    def define_objective_function(self, solver: Solver, variables, request) -> Solver:
        raise NotImplementedError

    @abstractmethod
    def define_constraints(self, solver: Solver, variables, request) -> Solver:
        raise NotImplementedError

    @abstractmethod
    def init_variables(self, solver: Solver, request) -> tuple[Solver, "Variables"]:
        raise NotImplementedError

    @abstractmethod
    def get_schedule_from_solution(self, variables, request) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def construct_request(self, forecast: pd.DataFrame) -> "Request":
        raise NotImplementedError

    @abstractmethod
    def update_state(self, schedule: pd.DataFrame) -> Any:
        """
        Updates the current state of the system based on the schedule.
        This method should be implemented in subclasses to define how the state is updated.
        """
        raise NotImplementedError
