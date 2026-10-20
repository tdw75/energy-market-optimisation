from dataclasses import dataclass

import pandas as pd
from ortools.linear_solver.pywraplp import Variable as ORToolsVariable
from ortools.linear_solver.pywraplp import Solver


@dataclass
class Variables:
    state: dict[pd.Timestamp, ORToolsVariable]
    charge: dict[pd.Timestamp, ORToolsVariable]
    discharge: dict[pd.Timestamp, ORToolsVariable]


def init_variables(
    solver: Solver, time_steps: list[pd.Timestamp]
) -> tuple[Solver, Variables]:

    def init_var(ts, name: str ):
        return solver.NumVar(
            0, solver.infinity(), f"{name}-{ts.strftime('%Y-%m-%d %X')}"
        )

    state = {t: init_var(t, "state") for t in time_steps}
    charge = {t: init_var(t, "charge") for t in time_steps}
    discharge = {t: init_var(t, "discharge") for t in time_steps}

    return solver, Variables(state=state, charge=charge, discharge=discharge)
