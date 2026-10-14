from dataclasses import dataclass
from ortools.linear_solver.pywraplp import Variable as ORToolsVariable
from ortools.linear_solver.pywraplp import Solver


@dataclass
class Variables:
    state: dict[int, ORToolsVariable]
    charge: dict[int, ORToolsVariable]
    discharge: dict[int, ORToolsVariable]


def init_variables(solver: Solver, time_steps: list[int]) -> tuple[Solver, Variables]:

    state = {t: solver.NumVar(0, solver.infinity(), f"state-{t}") for t in time_steps}
    charge = {t: solver.NumVar(0, solver.infinity(), f"charge-{t}") for t in time_steps}
    discharge = {t: solver.NumVar(0, solver.infinity(), f"discharge-{t}") for t in time_steps}

    return solver, Variables(state=state, charge=charge, discharge=discharge)
