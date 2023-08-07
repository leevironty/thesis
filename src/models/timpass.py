from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    LpInteger,
    LpBinary,
    LpAffineExpression,
    lpSum,
)
from thesis.data.wrapper import Data, Solution
from models.solver import get_solver
from thesis.data.schema import Event

pair = tuple[int, int]


class TimPass:
    def __init__(self, data: Data):
        self.data = data
        self.model = LpProblem(
            name='TimPass',
            sense=LpMinimize,
        )
        self.var_pi: dict[int, LpVariable] = LpVariable.dict(
            name='pi',
            indices=data.events.keys(),
            lowBound=0,
            upBound=data.config.period_length - 1,
            cat=LpInteger
        )
        self.var_z: dict[pair, LpVariable] = LpVariable.dicts(
            name='z',
            indices=data.activities_constrainable.keys(),
            cat=LpInteger
        )
        self.var_p: dict[pair, dict[pair, LpVariable]] = LpVariable.dicts(
            name='p',
            indices=(data.ods.keys(), data.activities.keys()),
            cat=LpBinary,
        )
        self.var_lin: dict[pair, dict[pair, LpVariable]] = LpVariable.dicts(
            name='lin',
            indices=(data.ods.keys(), data.activities.keys()),
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
        big_m = max([
            activity.upper_bound + activity.penalty
            for activity in data.activities.values()
        ])
        self.constraint_lin_mp = {
            f'lin_mp_{od}_{a}': lin <= big_m * self.var_p[od][a]
            for od, _v in self.var_lin.items() for a, lin in _v.items()
        }
        self.constraint_lin_duration = {
            f'lin_d_{od}_{a}': lin <= self._edge_penalty(a)
            for od, _v in self.var_lin.items() for a, lin in _v.items()
        }
        self.constraint_lin_duration_mp = {
            f'lin_d_mp_{od}_{a}': (
                lin >= self._edge_penalty(a) - big_m * (1 - self.var_p[od][a])
            )
            for od, _v in self.var_lin.items() for a, lin in _v.items()
        }

        # Constraint 8: path start and endpoints
        def d(i: int, j: int) -> int:
            return 1 if i == j else 0

        outbound: dict[pair, dict[int, list[LpVariable]]] = {}
        inbound: dict[pair, dict[int, list[LpVariable]]] = {}
        # outbound_aux: dict[pair, dict[int, list[LpVariable]]] = {}
        # inbound_aux: dict[pair, dict[int, list[LpVariable]]] = {}
        print(f'Origin events: {self.data.events_aux[0].keys()}')
        print(f'Destination events: {self.data.events_aux[1].keys()}')
        # self.constraint_useless_flows: dict[str, LpAffineExpression] = {}
        for uv, _p in self.var_p.items():
            for (from_node, to_node), p in _p.items():
                # if from_node in uv or to_node in uv:
                #     out_dict = outbound_aux
                #     in_dict = inbound_aux
                # else:
                # out_dict = outbound
                # in_dict = inbound
                # from_ok = from_node in uv or from_node in self.data.events.keys()
                # to_ok = to_node in uv or to_node in self.data.events.keys()
                # if not (from_ok and to_ok):
                #     print(f'Dropped flow variable: {uv=}, {(from_node, to_node)=}, {from_ok=}, {to_ok=}')
                #     self.constraint_useless_flows[f'useless_flow_{uv}_{(from_node, to_node)}'] = p == 0
                #     continue
                outbound.setdefault(uv, {}).setdefault(from_node, []).append(p)
                inbound.setdefault(uv, {}).setdefault(to_node, []).append(p)

        # self.constraint_paths = {
        #     f'paths_(origin={u},destination={v},event={e})': (
        #         lpSum(outbound[(u, v)].get(e, []))
        #         - lpSum(inbound[(u, v)].get(e, []))
        #         == d(u, e) - d(v, e)
        #     )
        #     for (u, v) in self.var_p.keys() for e in data.events_all.keys()
        # }

        self.constraint_paths: dict[str, LpAffineExpression] = {}
        self.constraint_od_flow: dict[str, LpAffineExpression] = {}
        for (origin, destination) in self.var_p.keys():
            origin_flow: list[LpVariable] = []
            destination_flow: list[LpVariable] = []
            def get_rhs(event: Event) -> LpVariable | int:
                if event.stop_id == origin:
                    var = LpVariable(
                        name=f'origin_flow_{(origin, destination)}_{event.event_id}',
                        lowBound=0,
                        upBound=1,
                        cat=LpBinary,
                    )
                    origin_flow.append(var)
                    return var
                if event.stop_id == destination:
                    var = LpVariable(
                        name=f'destination_flow_{(origin, destination)}_{event.event_id}',
                        lowBound=0,
                        upBound=1,
                        cat=LpBinary,
                    )
                    destination_flow.append(var)
                    return -var
                return 0
            for e, event in self.data.events.items():
                outbound_sum = lpSum(outbound[(origin, destination)].get(e, []))
                inbound_sum = lpSum(inbound[(origin, destination)].get(e, []))
                key = f'path_{(origin, destination)}_{e}'
                self.constraint_paths[key] = outbound_sum - inbound_sum == get_rhs(event)
                # self.constraint_paths[key] = outbound_sum - inbound_sum == 0
            # key = f'origin_flow_sum_{(origin, destination)}'
            # self.constraint_od_flow[key] = lpSum(origin_flow) == 1
            # key = f'destination_flow_sum_{(origin, destination)}'
            # self.constraint_od_flow[key] = lpSum(destination_flow) == 1



        # Handle preprocessed events: set all flows to zero
        if data.preprocessed_flows is not None:
            self.constraint_preprocessed = {
                f'preprocess_{uv}_{ij}': self.var_p[uv][ij] == 0
                for uv, _list in data.preprocessed_flows.items() for ij in _list
            }
        else:
            self.constraint_preprocessed = {}

        self.objective = self._get_objective()

        self.model.setObjective(self.objective)
        self.model.extend(self.constraint_lower)
        self.model.extend(self.constraint_upper)
        self.model.extend(self.constraint_lin_mp)
        self.model.extend(self.constraint_lin_duration)
        self.model.extend(self.constraint_lin_duration_mp)
        self.model.extend(self.constraint_paths)
        self.model.extend(self.constraint_preprocessed)
        self.model.extend(self.constraint_paths)
        self.model.extend(self.constraint_od_flow)
        # self.model.extend(self.constraint_useless_flows)
        # self.model.extend(self.constraint_aux_flows)

    def _get_objective(self) -> LpAffineExpression:
        return lpSum([
            od.customers * lpSum([
                self.var_lin[uv][a]
                for a in self.data.activities.keys()
            ])
            for uv, od in self.data.ods.items()
        ])

    def _edge_penalty(self, a: pair) -> LpAffineExpression:
        return self._edge_duration(a) + self.data.activities[a].penalty

    def _edge_duration(self, a: pair) -> LpAffineExpression:
        i, j = a
        return (
            self.var_pi[j] - self.var_pi[i]
            + self.var_z[a] * self.data.config.period_length
        )

    def solve(self):
        solver = get_solver()
        self.model.solve(solver)

    def get_solution(self) -> Solution:
        def result(var: LpVariable) -> int:
            # for type checking of the timetable values
            value = var.value()
            if value is None:
                raise ValueError('Encountered None in solution values!')
            return int(value)

        timetable = {
            event_id: result(pi)
            for event_id, pi in self.var_pi.items()
        }
        used_edges = [
            (uv, ij)
            for uv, _var in self.var_p.items() for ij, p in _var.items()
            if p.value() == 1
        ]
        weights: dict[pair, int] = {}
        for uv, ij in used_edges:
            prev = weights.get(ij, 0)
            weights[ij] = prev + self.data.ods[uv].customers

        return Solution(
            timetable=timetable,
            used_edges=used_edges,
            weights=weights,

        )

    def print_solution(self):
        timetable = {
            event_id: pi.value()
            for event_id, pi in self.var_pi.items()
        }
        used_edges = {
            (uv, ij): p.value()
            for uv, _var in self.var_p.items() for ij, p in _var.items()
        }
        print('Timetable:')
        for key, value in timetable.items():
            print(f'{key}: {value}')
        print('Used edges:')
        for key, value in used_edges.items():
            print(f'{key}: {value}')

    def print_constraints(self):
        print('Constraints:')
        for key, constraint in self.model.constraints.items():
            print(f'{key}: {constraint}')

    def print_objective(self):
        print('Objective:')
        print(self.model.objective)

    # def print_definition_counts(self):

    #     print('Variables:')
    #     print(f'count (pi) = {len(self.var_pi)}')
    #     print(f'count (p) = {len(self.var_p)}')
    #     print(f'count (z) = {len(self.var_z)}')
    #     print(f'count (lin) = {sum(len(d) for d in self.var_l.values())}')
    #     print('Constraints')
    #     print(f'count (l_d) = {len(self.)}')
