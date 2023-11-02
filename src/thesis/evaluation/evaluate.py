from pulp import LpSolver

from thesis.models.mip.pesp import PESP
from thesis.data.wrapper import Data
import networkx as nx

from thesis.models.mip.preprocessor import preprocess

pair = tuple[int, int]


def evaluate_weights(
    data: Data, weights: dict[tuple[int, int], float], solver: LpSolver, allow_non_optimal: bool = False
) -> float:
    model = PESP(
        data=data,
        weights=weights,
        solver=solver,
    )
    model.solve()
    durations = model.get_durations(allow_non_optimal)
    graph = nx.DiGraph()
    graph.add_edges_from(data.activities_routable)

    def weight(u: int, v: int, _) -> int:
        return durations.get((u, v), 0) + data.activities_routable[(u, v)].penalty

    # check for preprocessed routes being used in the evaluation
    assert data.preprocessed_flows is not None
    preprocess = data.preprocessed_flows

    # norm_weights: dict[pair, float] = {}
    objective = 0
    for (u, v), od in data.ods_mapped.items():
        shortest_path = nx.shortest_path(graph, u, v, weight=weight)
        for edge in zip(shortest_path[:-1], shortest_path[1:]):
            if edge in preprocess[(u, v)]:
                raise RuntimeError('Something is not correct, tried to use a preprocessed edge')
        length = nx.shortest_path_length(graph, u, v, weight=weight)
        objective += length * od.customers

    return objective
