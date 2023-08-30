from pulp import LpSolver

from thesis.models.mip.pesp import PESP
from thesis.data.wrapper import Data
import networkx as nx

pair = tuple[int, int]


def evaluate_weights(
    data: Data, weights: dict[tuple[int, int], float], solver: LpSolver
) -> float:
    model = PESP(
        data=data,
        weights=weights,
        solver=solver,
    )
    model.solve()
    durations = model.get_durations()
    graph = nx.DiGraph()
    graph.add_edges_from(data.activities_routable)

    def weight(u: int, v: int, _) -> int:
        return durations.get((u, v), 0) + data.activities_routable[(u, v)].penalty

    # norm_weights: dict[pair, float] = {}
    objective = 0
    for (u, v), od in data.ods_mapped.items():
        length = nx.shortest_path_length(graph, u, v, weight=weight)
        objective += length * od.customers

    return objective
