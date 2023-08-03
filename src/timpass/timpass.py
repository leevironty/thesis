from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpBinary, LpConstraint, LpAffineExpression, lpSum, lpDot
from thesis.data.wrapper import Data


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
        self.var_z: dict[tuple[int, int], LpVariable] = LpVariable.dict(
            name='z',
            indices=data.activities.keys(),
            cat=LpInteger
        )
        self.var_p: dict[tuple[int, int], dict[tuple[int, int], LpVariable]] = LpVariable.dicts(
            name='p',
            indices=(data.od.keys(), data.activities_all.keys()),
            cat=LpBinary,
        )
        self.var_l: dict[tuple[int, int], dict[tuple[int, int], LpVariable]] = LpVariable.dicts(
            name='linearization_p_duration',
            indices=(data.od.keys(), data.activities.keys()),
            lowBound=0,
            cat=LpInteger,
        )
        self.constraint_lower = {
            a: self._get_edge_duration(a) >= activity.lower_bound
            for a, activity in data.activities.items()
        }
        self.constraint_upper = {
            a: self._get_edge_duration(a) <= activity.upper_bound
            for a, activity in data.activities.items()
        }
        # linearization
        big_m = max([
            activity.upper_bound for activity in data.activities_all.values()
        ])
        self.constraint_l_mp = {
            (od, a): lin <= big_m * self.var_p[od][a]
            for od, _v in self.var_l.items() for a, lin in _v.items()
        }
        self.constraint_l_d = {
            (od, a): lin <= self._get_edge_duration(a)
            for od, _v in self.var_l.items() for a, lin in _v.items()
        }
        self.constraint_l_d_mp = {
            (od, a): lin >= self._get_edge_duration(a) - big_m * (1 - self.var_p[od][a])
            for od, _v in self.var_l.items() for a, lin in _v.items()
        }

        # Constraint 8: path start and endpoints
        def d(i: int, j: int) -> int:
            return 1 if i == j else 0

        outbound: dict[int, list[LpVariable]] = {}
        inbound: dict[int, list[LpVariable]] = {}
        for (u, v), _p in self.var_p.items():
            for p in _p.values():
                outbound.setdefault(u, []).append(p)
                inbound.setdefault(v, []).append(p)

        self.constraint_paths = {
            ((u, v), e): (
                lpSum(outbound[e]) - lpSum(inbound[e])
                == d(u, e) - d(v, e)
            )
            for (u, v) in self.var_p.keys() for e in data.events_all.keys()
        }

        self.objective = self._get_objective()

        self.model.setObjective(self.objective)
        self.model.extend(self.constraint_lower)
        self.model.extend(self.constraint_upper)
        self.model.extend(self.constraint_l_mp)
        self.model.extend(self.constraint_l_d)
        self.model.extend(self.constraint_l_d_mp)
        self.model.extend(self.constraint_paths)

    def _get_objective(self) -> LpAffineExpression:
        return lpSum([
            od.customers * lpSum([
                self.var_l[uv][a]
                for a in self.data.activities.keys()
            ])
            for uv, od in self.data.od.items()
        ])

    def _get_edge_duration(self, a: tuple[int, int]) -> LpAffineExpression:
        i, j = a
        return (
            self.var_pi[j] - self.var_pi[i]
            + self.var_z[a] * self.data.config.period_length
        )
