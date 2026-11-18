from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from ortools.linear_solver import pywraplp

from src.data.raw import Resolution


class BaseORToolsLP(ABC):
    """Base class for OR-Tools linear programming models."""

    def __init__(self, config, current_state: Any, resolution: Resolution):
        self.config = config
        self.current_state = current_state
        self.resolution = resolution
        self.solver = pywraplp.Solver.CreateSolver(
            "GLOP"
        )  # todo: make solver type configurable

    @abstractmethod
    def solve(self, forecast: pd.Series) -> Any:
        raise NotImplementedError

    @abstractmethod
    def define_objective_function(self, variables, request):
        raise NotImplementedError

    @abstractmethod
    def define_constraints(self, variables, request):
        raise NotImplementedError

    @abstractmethod
    def init_variables(self, request) -> "Variables":
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
