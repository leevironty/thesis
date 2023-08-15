from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    LpInteger,
    LpAffineExpression,
    lpSum,
    LpSolver,
)
from thesis.data.wrapper import Data

# from models.solver import get_solver
import networkx as nx


pair = tuple[int, int]


class PESP:
    def __init__(self, data: Data, weights: dict[pair, float], solver: LpSolver):
        self.data = data
        self.solver = solver
        self.weights = weights

        graph = nx.Graph()
        graph.add_edges_from(data.activities_constrainable)
        cycles: list[list[int]] = nx.cycle_basis(graph)

        self.model = LpProblem(
            name='TimPass',
            sense=LpMinimize,
        )
        self.var_z: dict[int, LpVariable] = LpVariable.dicts(
            name='z', indices=list(range(len(cycles))), cat=LpInteger
        )
        self.var_x: dict[pair, LpVariable] = LpVariable.dicts(
            name='x',
            indices=(data.activities_constrainable.keys()),
            cat=LpInteger,
        )
        self.constraint_lower = {
            f'duration_lb_{a}': self._edge_duration(a) >= activity.lower_bound
            for a, activity in data.activities_constrainable.items()
        }
        self.constraint_upper = {
            f'duration_ub_{a}': self._edge_duration(a) <= activity.upper_bound
            for a, activity in data.activities_constrainable.items()
        }

        self.constraint_cycle: dict[str, LpAffineExpression] = {}
        for index, cycle in enumerate(cycles):
            rhs = 0
            lhs = self.var_z[index] * self.data.config.period_length
            for i, j in zip(cycle, cycle[1:] + [cycle[0]]):
                if (i, j) in self.data.activities_constrainable:
                    rhs += self.var_x[(i, j)]
                else:
                    rhs -= self.var_x[(j, i)]
            key = f'cycle_{index}'
            self.constraint_cycle[key] = lhs == rhs

        self.objective = self._get_objective()

        self.model.setObjective(self.objective)
        self.model.extend(self.constraint_lower)
        self.model.extend(self.constraint_upper)
        self.model.extend(self.constraint_cycle)

    def _get_objective(self) -> LpAffineExpression:
        return lpSum(
            [self._edge_penalty(ij) * weight for ij, weight in self.weights.items()]
        )

    def _edge_penalty(self, a: pair) -> LpAffineExpression:
        return self._edge_duration(a) + self.data.activities[a].penalty

    def _edge_duration(self, ij: pair) -> LpVariable | LpAffineExpression:
        return self.var_x[ij]

    def solve(self):
        self.model.solve(self.solver)

    def print_constraints(self):
        print('Constraints:')
        for key, constraint in self.model.constraints.items():
            print(f'{key}: {constraint}')

    def print_objective(self):
        print('Objective:')
        print(self.model.objective)
