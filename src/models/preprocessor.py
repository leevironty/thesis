from typing import TypeVar
from functools import partial
import math

import networkx as nx

from thesis.data.wrapper import Data
from thesis.data.schema import Activity, Edge

T = TypeVar('T')

pair = tuple[T, T]


def weight_worst_case(u: int, v: int, _, edges: dict[pair[int], Edge]) -> int:
    edge = edges[(u, v)]
    return edge.upper_bound


def weight_best_case(u: int, v: int, _, edges: dict[pair[int], Edge]) -> int:
    edge = edges[(u, v)]
    return edge.lower_bound


def preprocess(data: Data) -> dict[pair[int], list[pair[int]]]:
    graph = nx.DiGraph()
    graph.add_edges_from(data.activities_routable)
    w_worst = partial(weight_worst_case, edges=data.activities_routable)
    w_best = partial(weight_best_case, edges=data.activities_routable)
    beta: dict[int, dict[int, int]] = dict(nx.shortest_path_length(graph, weight=w_worst))
    best_case: dict[int, dict[int, int]] = dict(nx.shortest_path_length(graph, weight=w_best))

    out: dict[pair[int], list[pair[int]]] = {}
    for (u, v), od in data.ods_mapped.items():
        for (i, j), activity in data.activities_routable.items():
            gamma = best_case[u].get(i, math.inf)
            sigma = best_case[j].get(v, math.inf)
            if gamma + activity.lower_bound + sigma > beta[u][v]:
                out.setdefault((od.origin, od.destination), []).append((i, j))

    return out
