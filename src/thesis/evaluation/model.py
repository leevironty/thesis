from abc import ABC, abstractmethod
from thesis.data.wrapper import Data
import networkx as nx


pair = tuple[int, int]


class Model(ABC):
    @abstractmethod
    def __init__(self, data: Data):
        raise NotImplementedError

    @abstractmethod
    def get_weights(self) -> dict[pair, float]:
        raise NotImplementedError


class Baseline(Model):
    def __init__(self, data: Data):
        self.data = data

    def get_weights(self) -> dict[pair, float]:
        def get_distance(i: int, j: int, _) -> int:
            activity = self.data.activities_routable[(i, j)]
            return activity.lower_bound + activity.penalty

        graph = nx.DiGraph()
        graph.add_edges_from(self.data.activities_routable)

        weights: dict[pair, float] = {}
        for (u, v), od in self.data.ods_mapped.items():
            shortest_path: list[int] = nx.shortest_path(
                graph,
                u,
                v,
                weight=get_distance,
            )
            first = shortest_path[1:-2]
            second = shortest_path[2:-1]
            for ij in zip(first, second):
                # if ij not in self.data.activities:
                #     ij = ij[1], ij[0]
                #     if ij not in self.data.activities:
                #         raise RuntimeError(
                #             'Could not find path edge in activities',
                #         )
                value = weights.get(ij, 0.0)
                weights[ij] = value + float(od.customers)
        for ij in self.data.activities.keys():
            if ij not in weights:
                weights[ij] = 0.0
        return weights
