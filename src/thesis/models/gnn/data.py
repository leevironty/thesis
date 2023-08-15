import pathlib

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from thesis.data.wrapper import Data as ThesisData
from thesis.data.schema import (
    Event,
    EventType,
    Activity,
    ActivityType,
    Direction,
    OD,
)


def get_data(path: str) -> list[HeteroData]:
    paths = list(pathlib.Path(path).glob('**/solution_*.pkl.gz'))
    return [transform(ThesisData.from_pickle(file.as_posix())) for file in paths]


def get_loaders(
    path: str, threads: int = 1, batch_size: int = 1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    all_data = get_data(path)
    train_loader = DataLoader(
        dataset=all_data[:-100],
        batch_size=batch_size,
        shuffle=True,
        num_workers=threads,
    )
    val_loader = DataLoader(
        dataset=all_data[-100:-50],
        batch_size=batch_size,
        num_workers=threads,
    )
    test_loader = DataLoader(
        dataset=all_data[-50:],
        batch_size=batch_size,
        num_workers=threads,
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
    out['target'].edge_index = (
        torch.tensor(
            [
                [event_map[right], event_map[left]]
                for left, right in data.activities.keys()
            ],
            dtype=torch.int64,
        )
        .t()
        .contiguous()
    )
    out['target'].weight = torch.tensor(
        [
            data.solution.weights.get((left, right), 0.0)
            for left, right in data.activities.keys()
        ],
        dtype=torch.float,
    )

    return out
