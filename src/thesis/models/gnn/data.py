import pathlib
import logging

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.transforms import AddMetaPaths


from thesis.data.wrapper import Data as ThesisData
from thesis.evaluation.model import get_trivial_weights
from thesis.data.schema import (
    Event,
    EventType,
    Activity,
    ActivityType,
    Direction,
    OD,
)


logger = logging.getLogger(__name__)


class SolutionDataset(Dataset):
    def __init__(self, filenames: list[str]):
        self._paths = filenames
        super().__init__()

    @property
    def processed_file_names(self):
        return self._paths
    
    def get(self, idx: int) -> HeteroData:
        return torch.load(self._paths[idx])

    def len(self) -> int:
        return len(self._paths)



class TimPassDataset(Dataset):
    def __init__(self, paths: list[pathlib.Path]):
        super().__init__()
        self.paths = paths
        self.in_memory_cache: dict[int, BaseData] = {}

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> BaseData:
        if idx in self.in_memory_cache:
            return self.in_memory_cache[idx]
        filename = self.paths[idx]
        cached_filename = filename.with_suffix('.pt')
        if cached_filename.exists():
            out = torch.load(cached_filename)
            self.in_memory_cache[idx] = out
            return out
        out = transform(ThesisData.from_pickle(filename.as_posix()))
        torch.save(out, cached_filename)
        self.in_memory_cache[idx] = out
        return out


def get_data(path: str) -> list[HeteroData]:
    paths = sorted(list(pathlib.Path(path).glob('**/solution_*.pkl.gz')))

    return [transform(ThesisData.from_pickle(file.as_posix())) for file in paths]


# def get_data_iter(files: Iterable[pathlib.Path]) -> Iterable[HeteroData]:
#     for file in files:
#         yield transform(ThesisData.from_pickle(file.as_posix()))


# def get_loaders(
#     path: str, threads: int = 1, batch_size: int = 1
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     all_data = get_data(path)
#     train_loader = DataLoader(
#         dataset=all_data[:-100],
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=threads,
#     )
#     val_loader = DataLoader(
#         dataset=all_data[-100:-50],
#         batch_size=batch_size,
#         num_workers=threads,
#     )
#     test_loader = DataLoader(
#         dataset=all_data[-50:],
#         batch_size=batch_size,
#         num_workers=threads,
#     )
#     return train_loader, val_loader, test_loader


def get_file(path: pathlib.Path) -> HeteroData:
    cached_filename = path.with_suffix('.pt')
    if cached_filename.exists():
        out = torch.load(cached_filename)
        return out
    out = transform(ThesisData.from_pickle(path.as_posix()))
    torch.save(out, cached_filename)
    return out


def get_data_list_no_transform(paths: list[pathlib.Path]) -> list[HeteroData]:
    return [get_file(path) for path in paths]


def get_dataset(paths: list[pathlib.Path]) -> Dataset:
    filenames = [path.as_posix() for path in paths]
    return SolutionDataset(filenames)


def get_loaders(
    path: str, threads: int = 1, batch_size: int = 1, test_share: float = 0.1, val_share: float = 0.1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    assert test_share + val_share < 1.0, 'Sum of shares must be below one!'
    # paths = list(pathlib.Path(path).glob('**/solution_*.pkl.gz'))
    logger.debug('Listing file paths')
    paths = sorted(list(pathlib.Path(path).glob('**/solution_*.pt')))
    test_index = int((test_share + val_share) * len(paths))
    val_index = int(val_share * len(paths))
    # test_dataset = TimPassDataset(paths[:-test_index])
    logger.debug('Getting train loader')
    train_dataset = get_dataset(paths[:-test_index])
    logger.debug('Getting val and test loaders')
    val_dataset = get_dataset(paths[-test_index:-val_index])
    test_dataset = get_dataset(paths[-val_index:])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=threads,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        # dataset=TimPassDataset(paths[-test_index:-val_index]),
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=threads,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        # dataset=TimPassDataset(paths[-val_index:]),
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=threads,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader


def transform_event(event: Event) -> list[float]:
    type_enc = [
        float(event.type == EventType.ARRIVAL),
        float(event.type == EventType.DEPARTURE),
    ]
    dir_enc = [
        float(event.line_direction == Direction.FORWARDS),
        float(event.line_direction == Direction.BACKWARDS),
    ]
    freq_enc = [float(event.line_freq_repetition)]
    return type_enc + dir_enc + freq_enc


def transform_activity(activity: Activity, period: int) -> list[float]:
    type_enc = [
        float(activity.type == ActivityType.DRIVE),
        float(activity.type == ActivityType.WAIT),
        float(activity.type == ActivityType.CHANGE),
        float(activity.type == ActivityType.HEADWAY),
        float(activity.type == ActivityType.SYNC),
    ]
    time_enc = [
        activity.lower_bound / period,
        activity.upper_bound / period,
        activity.penalty / period,
    ]
    return type_enc + time_enc


def transform_od(od: OD, mean: float) -> list[float]:
    return [od.customers / mean]


def transform(data: ThesisData) -> HeteroData:
    assert data.preferences is not None
    out = HeteroData()
    event_map: dict[int, int] = dict(
        (key, i) for i, key in enumerate(data.events.keys())
    )
    stop_map: dict[int, int] = dict(
        (key, i)
        for i, key in enumerate(set(event.stop_id for event in data.events.values()))
    )
    line_map: dict[int, int] = dict(
        (key, i)
        for i, key in enumerate(set(event.line_id for event in data.events.values()))
    )
    # 1. routing parts
    # 2. stops, ods, event to stop link
    # 3. line ids
    # 4. top n paths
    # remember: make it easy to include positional / identity encodings

    # nodes
    out['event'].x = torch.tensor(
        [transform_event(event) for event in data.events.values()], dtype=torch.float
    )
    out['stop'].x = torch.zeros((len(stop_map), 1), dtype=torch.float)
    # TODO: include line freq / total
    out['line'].x = torch.zeros((len(line_map), 1), dtype=torch.float)

    stop_event_relation = (
        torch.tensor(
            [
                [event_map[key], stop_map[event.stop_id]]
                for key, event in data.events.items()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )

    out['event', 'belongs', 'stop'].edge_index = stop_event_relation
    out['stop', 'has', 'event'].edge_index = stop_event_relation[[1, 0], :].contiguous()

    event_line_relation = (
        torch.tensor(
            [
                [event_map[key], line_map[event.line_id]]
                for key, event in data.events.items()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    out['event', 'belongs', 'line'].edge_index = event_line_relation
    out['line', 'has', 'event'].edge_index = event_line_relation[[1, 0], :].contiguous()

    # handle routing edges
    routes: list[tuple[int, int]] = []
    event_route_relation: list[tuple[int, int]] = []
    route_features: list[list[float]] = []
    target_mask: list[int] = []
    if data.solution is not None:
        target_values: list[float] = []
    trivial_weights = get_trivial_weights(data)
    max_trivial_weight = max(trivial_weights.values())
    for i, ((u, v), activity) in enumerate(data.activities_constrainable.items()):
        preference = data.preferences.get((u, v), 1.0)
        features = transform_activity(activity, data.config.period_length)
        norm_trivial_weight = trivial_weights.get((u, v), 0) / max_trivial_weight
        features += [preference, norm_trivial_weight]
        route_features.append(features)
        routes.append((event_map[u], event_map[v]))
        routes.append((event_map[v], event_map[u]))
        event_route_relation += [(event_map[u], i), (event_map[v], i)]
        target_mask.append(1 if (u, v) in data.activities else 0)
        if data.solution is not None:
            target_values.append(data.solution.weights.get((u, v), 0.0))

    out['route_features'].x = torch.tensor(
        route_features,
        dtype=torch.float32,
    )
    out['route_features'].mask = torch.tensor(
        target_mask,
        dtype=torch.float32,
    ).reshape(-1, 1)
    if data.solution is not None:
        out['route_features'].target = torch.tensor(
            target_values,
            dtype=torch.float32,
        ).reshape(-1, 1)
        out['route_features'].target /= out['route_features'].target.max()
    relation = torch.tensor(event_route_relation, dtype=torch.int64).t().contiguous()
    out['event', 'connects', 'route_features'].edge_index = relation
    out['route_features', 'connects', 'event'].edge_index = relation[
        [1, 0], :
    ].contiguous()
    out['event', 'routes', 'event'].edge_index = (
        torch.tensor(routes, dtype=torch.int64).t().contiguous()
    )

    ods: list[tuple[int]] = []
    stop_od_origin: list[tuple[int, int]] = []
    stop_od_destination: list[tuple[int, int]] = []
    stop_stop: list[tuple[int, int]] = []
    for i, ((u, v), od) in enumerate(data.ods.items()):
        ods.append((od.customers,))
        stop_od_origin.append((stop_map[u], i))
        stop_od_destination.append((stop_map[v], i))
        stop_stop += [(stop_map[u], stop_map[v]), (stop_map[v], stop_map[u])]
    ods_feature = torch.tensor(ods, dtype=torch.float32)
    ods_feature /= ods_feature.max()
    out['od'].x = ods_feature

    stop_od_origin_tensor = (
        torch.tensor(stop_od_origin, dtype=torch.int64).t().contiguous()
    )
    stop_od_destination_tensor = (
        torch.tensor(stop_od_destination, dtype=torch.int64).t().contiguous()
    )
    stop_stop_tensor = torch.tensor(stop_stop, dtype=torch.int64).t().contiguous()
    out['stop', 'origin', 'od'].edge_index = stop_od_origin_tensor
    out['od', 'origin', 'stop'].edge_index = stop_od_origin_tensor[
        [1, 0], :
    ].contiguous()
    out['stop', 'destination', 'od'].edge_index = stop_od_destination_tensor
    out['od', 'destination', 'stop'].edge_index = stop_od_destination_tensor[
        [1, 0], :
    ].contiguous()
    out['stop', 'linked', 'stop'].edge_index = stop_stop_tensor

    metapaths = [
        [('od', 'origin', 'stop'), ('stop', 'event'), ('event', 'route_features')],
        [('od', 'destination', 'stop'), ('stop', 'event'), ('event', 'route_features')],
        [('route_features', 'event'), ('event', 'route_features')],
    ]
    out = AddMetaPaths(metapaths)(out)

    return out


def transform_old(data: ThesisData) -> HeteroData:
    out = HeteroData()
    event_map: dict[int, int] = dict(
        (key, i) for i, key in enumerate(data.events.keys())
    )
    stop_map: dict[int, int] = dict(
        (key, i)
        for i, key in enumerate(set(event.stop_id for event in data.events.values()))
    )

    # nodes
    out['event'].x = torch.tensor(
        [transform_event(event) for event in data.events.values()], dtype=torch.float
    )
    out['stop'].x = torch.tensor([[0] for _ in stop_map], dtype=torch.float)

    # edges
    out['stop', 'demand', 'stop'].edge_index = (
        torch.tensor(
            data=[
                [stop_map[od.origin], stop_map[od.destination]]
                for od in data.ods.values()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    od_mean = sum(od.customers for od in data.ods.values()) / len(data.ods)
    out['stop', 'demand', 'stop'].edge_attr = torch.tensor(
        [transform_od(od, mean=od_mean) for od in data.ods.values()], dtype=torch.float
    )

    out['event', 'belongs', 'stop'].edge_index = (
        torch.tensor(
            [
                [event_map[key], stop_map[event.stop_id]]
                for key, event in data.events.items()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    out['stop', 'has', 'event'].edge_index = (
        torch.tensor(
            [
                [stop_map[event.stop_id], event_map[key]]
                for key, event in data.events.items()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    out['event', 'belongs', 'stop'].edge_attr = torch.tensor(
        [[0] for _ in data.events], dtype=torch.float
    )
    out['stop', 'has', 'event'].edge_attr = torch.tensor(
        [[0] for _ in data.events], dtype=torch.float
    )

    out['event', 'routes', 'event'].edge_index = (
        torch.tensor(
            [
                [event_map[left], event_map[right]]
                for left, right in data.activities_constrainable.keys()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    out['event', 'routes', 'event'].edge_attr = torch.tensor(
        [
            transform_activity(activity, data.config.period_length)
            for activity in data.activities_constrainable.values()
        ],
        dtype=torch.float,
    )
    out['event', 'routes_reverse', 'event'].edge_index = (
        torch.tensor(
            [
                [event_map[right], event_map[left]]
                for left, right in data.activities_constrainable.keys()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    out['event', 'routes_reverse', 'event'].edge_attr = torch.tensor(
        [
            transform_activity(activity, data.config.period_length)
            for activity in data.activities_constrainable.values()
        ],
        dtype=torch.float,
    )
    if data.solution is None:
        return out
    out['event', 'target', 'event'].edge_index = torch.clone(
        out['event', 'routes', 'event'].edge_index
    )
    out['event', 'target', 'event'].edge_attr = torch.tensor(
        [
            [data.solution.weights.get((left, right), 0.0)]
            for left, right in data.activities.keys()
        ],
        dtype=torch.float,
    )
    # normalize to have weight mean of one
    out['event', 'target', 'event'].edge_attr /= out[
        'event', 'target', 'event'
    ].edge_attr.mean()
    # out['target'].edge_index = (
    #     torch.tensor(
    #         [
    #             [event_map[right], event_map[left]]
    #             for left, right in data.activities.keys()
    #         ],
    #         dtype=torch.int64,
    #     )
    #     .t()
    #     .contiguous()
    # # )
    # out['target'].weight = torch.tensor(
    #     [
    #         data.solution.weights.get((left, right), 0.0)
    #         for left, right in data.activities.keys()
    #     ],
    #     dtype=torch.float,
    # )

    return out
