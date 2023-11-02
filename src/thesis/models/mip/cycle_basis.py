from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    LpInteger,
    LpBinary,
    LpAffineExpression,
    lpSum,
    LpSolver,
)
from thesis.data.wrapper import Data, Solution

# from models.solver import get_solver
import networkx as nx


pair = tuple[int, int]


class TimPassCycle:
    def __init__(self, data: Data, solver: LpSolver):
        self.data = data
        self.solver = solver

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
        self.var_p: dict[pair, dict[pair, LpVariable]] = LpVariable.dicts(
            name='p',
            indices=(data.ods_mapped.keys(), data.activities_routable.keys()),
            cat=LpBinary,
        )
        self.var_l: dict[pair, dict[pair, LpVariable]] = LpVariable.dicts(
            name='lin',
            indices=(data.ods_mapped.keys(), data.activities.keys()),
            lowBound=0,
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
        # objective linearization
        big_m = max(
            [
                activity.upper_bound + activity.penalty
                for activity in data.activities.values()
            ]
        )
        self.constraint_l_mp = {
            f'lin_mp_{od}_{a}': lin <= big_m * self.var_p[od][a]
            for od, _v in self.var_l.items()
            for a, lin in _v.items()
        }
        self.constraint_l_d = {
            f'lin_d_{od}_{a}': lin <= self._edge_penalty(a)
            for od, _v in self.var_l.items()
            for a, lin in _v.items()
        }
        self.constraint_l_d_mp = {
            f'lin_d_mp_{od}_{a}': (
                lin >= self._edge_penalty(a) - big_m * (1 - self.var_p[od][a])
            )
            for od, _v in self.var_l.items()
            for a, lin in _v.items()
        }

        # Constraint 8: path start and endpoints
        def d(i: int, j: int) -> int:
            return 1 if i == j else 0

        outbound: dict[pair, dict[int, list[LpVariable]]] = {}
        inbound: dict[pair, dict[int, list[LpVariable]]] = {}
        # print(f'Origin events: {self.data.events_aux[0].keys()}')
        # print(f'Destination events: {self.data.events_aux[1].keys()}')
        # self.constraint_useless_flows: dict[str, LpAffineExpression] = {}
        for uv, _p in self.var_p.items():
            for (from_node, to_node), p in _p.items():
                outbound.setdefault(uv, {}).setdefault(from_node, []).append(p)
                inbound.setdefault(uv, {}).setdefault(to_node, []).append(p)

        self.constraint_paths = {
            f'paths_(origin={u},destination={v},event={e})': (
                lpSum(outbound[(u, v)].get(e, [])) - lpSum(inbound[(u, v)].get(e, []))
                == d(u, e) - d(v, e)
            )
            for (u, v) in self.var_p.keys()
            for e in data.events_all.keys()
        }

        # Handle preprocessed events: set all flows to zero
        if data.preprocessed_flows is not None:
            self.constraint_preprocessed = {
                f'preprocess_{uv}_{ij}': self.var_p[uv][ij] == 0
                for uv, _list in data.preprocessed_flows.items()
                for ij in _list
            }
        else:
            self.constraint_preprocessed = {}

        # cycle constraint
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
        self.model.extend(self.constraint_l_mp)
        self.model.extend(self.constraint_l_d)
        self.model.extend(self.constraint_l_d_mp)
        self.model.extend(self.constraint_paths)
        # self.model.extend(self.constraint_preprocessed)
        self.model.extend(self.constraint_cycle)

    def _get_objective(self) -> LpAffineExpression:
        return lpSum(
            [
                od.customers
                * lpSum([self.var_l[uv][a] for a in self.data.activities.keys()])
                for uv, od in self.data.ods_mapped.items()
            ]
        )

    def _edge_penalty(self, a: pair) -> LpAffineExpression:
        return self._edge_duration(a) + self.data.activities[a].penalty

    def _edge_duration(self, a: pair) -> LpVariable | LpAffineExpression:
        i, j = a
        return self.var_x[(i, j)]

    def solve(self):
        self.model.solve(self.solver)

    def get_solution(self) -> Solution:
        def result(var: LpVariable) -> int:
            # for type checking of the timetable values
            value = var.value()
            if value is None:
                raise ValueError('Encountered None in solution values!')
            return round(value)  # possible to encounter floating point rounding errors if this is not included

        # for uv, _var in self.var_p.items():
        #     for ij, p in _var.items():
        #         if not isinstance(p.value(), float):
        #             raise RuntimeError('Should not encounter non-float values')
        #         if p.value() != 0 and p.value() != 1:
        #             print(f'Found a rounding error: {uv, ij =}')

        used_edges = [
            (uv, ij)
            for uv, _var in self.var_p.items()
            for ij, p in _var.items()
            if p.value() > 0.5  # numerical tolerance?
            # if abs(p.value() - 1) < 0.001  # numerical tolerance
        ]
        # used_edges_old_accuracy_test = [
        #     (uv, ij)
        #     for uv, _var in self.var_p.items()
        #     for ij, p in _var.items()
        #     if p.value() == 1
        #     # if p.value() > 0.5  # numerical tolerance?
        #     # if abs(p.value() - 1) < 0.001  # numerical tolerance
        # ]
        weights: dict[pair, int] = {}
        for uv, ij in used_edges:
            prev = weights.get(ij, 0)
            weights[ij] = prev + self.data.ods_mapped[uv].customers
        
        # weights_old_accuracy_test: dict[pair, int] = {}
        # for uv, ij in used_edges_old_accuracy_test:
        #     prev = weights_old_accuracy_test.get(ij, 0)
        #     weights_old_accuracy_test[ij] = prev + self.data.ods_mapped[uv].customers

        # weights_robust: dict[pair, int] = {}
        # for uv, od in self.data.ods_mapped.items():
        #     for ij in self.data.activities.keys():
        #         if ij not in weights_robust:
        #             weights_robust[ij] = 0
        #         weights_robust[ij] += od.customers * self.var_p[uv][ij].value()

        edge_durations = {ij: result(var) for ij, var in self.var_x.items()}
        return Solution(
            # timetable=timetable,
            used_edges=used_edges,
            weights=weights,
            edge_durations=edge_durations,
            aux={
                # 'used_edges_old': used_edges_old_accuracy_test,
                # 'weights_old': weights_old_accuracy_test,
                # 'weights_robust': weights_robust,
                'x': {ij: var.value() for ij, var in self.var_x.items()},
                'z': {c: var.value() for c, var in self.var_z.items()},
                'p': {uv: {ij: var.value() for ij, var in d.items()} for uv, d in self.var_p.items()},
                'lin': {uv: {ij: var.value() for ij, var in d.items()} for uv, d in self.var_l.items()},
            }
        )

    # def print_solution(self):
    #     timetable = {
    #         event_id: pi.value()
    #         for event_id, pi in self.var_pi.items()
    #     }
    #     used_edges = {
    #         (uv, ij): p.value()
    #         for uv, _var in self.var_p.items() for ij, p in _var.items()
    #     }
    #     print('Timetable:')
    #     for key, value in timetable.items():
    #         print(f'{key}: {value}')
    #     print('Used edges:')
    #     for key, value in used_edges.items():
    #         print(f'{key}: {value}')

    def print_constraints(self):
        print('Constraints:')
        for key, constraint in self.model.constraints.items():
            print(f'{key}: {constraint}')

    def print_objective(self):
        print('Objective:')
        print(self.model.objective)
