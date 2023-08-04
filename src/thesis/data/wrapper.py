from typing import TypeVar
from attrs import define, field
import pandas as pd
import os
from enum import Enum
import pathlib

from thesis.data.schema import (
    Activity,
    Config,
    OD,
    Event,
    Node,
    Edge,
    ActivityType,
)

T = TypeVar('T')
pair = tuple[T, T]


# TODO: python 3.11 support StrEnum
class Filename(Enum):
    ACTIVITIES = "Activities.csv"
    CONFIG = "Config.csv"
    EVENTS = "Events.csv"
    OD = "OD.csv"
    TIMETABLE = 'timetable.csv'
    PATHS = 'used_paths.csv'
    WEIGHTS = 'weights.csv'


def read(path: str) -> pd.DataFrame:
    with open(path) as file:
        file.seek(2)  # Drop comment hashtags
        return pd.read_csv(file, sep=';').rename(columns=str.strip)


@define
class Solution:
    timetable: dict[int, int]
    used_edges: list[pair[pair[int]]]
    weights: dict[pair[int], int]
    # TODO: add meta, contains e.g. solver status, gap, used time,
    # objective value etc.


class _Counter:
    def __init__(self, init_value: int):
        self.value = init_value

    def next(self) -> int:
        self.value += 1
        return self.value


@define
class Data:
    config: Config
    activities: dict[pair[int], Activity]
    activities_aux: dict[pair[int], Edge] = field(init=False)
    activities_nonroutable: dict[pair[int], Activity] = field(init=False)
    events: dict[int, Event]
    events_aux: pair[dict[int, Node]] = field(init=False)
    ods: dict[pair[int], OD]
    ods_mapped: dict[pair[int], OD] = field(init=False)
    solution: Solution | None = field(default=None)

    def __attrs_post_init__(self):
        # TODO: refactor this mess
        # split activities by the activity type to routable and non-routable
        self.activities_nonroutable = {
            a: activity
            for a, activity in self.activities.items()
            if activity.type in [ActivityType.SYNC, ActivityType.HEADWAY]
        }
        self.activities = {
            a: activity
            for a, activity in self.activities.items()
            if activity.type not in [ActivityType.SYNC, ActivityType.HEADWAY]
        }
        od_node_id_maps = self._get_od_node_id_maps(self.events, self.ods)
        self.events_aux = self._get_events_aux(od_node_id_maps)
        self.activities_aux = self._get_activities_aux(
            self.events_aux,
            self.events,
            self.activities,
        )
        self.ods_mapped = self._get_ods_mapped(od_node_id_maps, self.ods)

    @staticmethod
    def _get_activities_aux(
        events_aux: pair[dict[int, Node]],
        events: dict[int, Event],
        activities: dict[pair[int], Activity],
    ) -> dict[pair[int], Edge]:
        max_edge_id = max(
            activity.activity_index for activity in activities.values()
        )
        next_id = _Counter(max_edge_id).next
        origin_events, destination_events = events_aux
        event_map: dict[int, list[Event]] = {}
        for event in events.values():
            event_map.setdefault(event.stop_id, []).append(event)
        # TODO: make more elegant
        aux_edges: dict[pair[int], Edge] = {}
        for origin_id, origin in origin_events.items():
            for event in event_map[origin.stop_id]:
                aux_edges[(origin_id, event.event_id)] = Edge(
                    activity_index=next_id(),
                    from_event=origin_id,
                    to_event=event.event_id,
                )
        for destination_id, destination in destination_events.items():
            for event in event_map[destination.stop_id]:
                aux_edges[(event.event_id, destination_id)] = Edge(
                    activity_index=next_id(),
                    from_event=event.event_id,
                    to_event=destination_id,
                )
        return aux_edges

    @property
    def activities_routable(self) -> dict[pair[int], Edge]:
        return self.activities | self.activities_aux

    @property
    def activities_constrainable(self) -> dict[pair[int], Activity]:
        return self.activities | self.activities_nonroutable

    @staticmethod
    def _get_od_node_id_maps(
        events: dict[int, Event],
        ods: dict[pair[int], OD],
    ) -> pair[dict[int, int]]:
        # origin & destination map: stop_id -> node_id
        max_event_id = max(events.keys())
        next_id = _Counter(max_event_id).next
        origins, destinations = zip(*ods.keys())
        origins = set(origins)
        destinations = set(destinations)
        origin_map: dict[int, int] = {
            origin: next_id() for origin in origins
        }
        destination_map: dict[int, int] = {
            destination: next_id() for destination in destinations
        }
        return origin_map, destination_map

    @staticmethod
    def _get_events_aux(
        od_node_id_maps: pair[dict[int, int]]
    ) -> pair[dict[int, Node]]:
        origin_map, destination_map = od_node_id_maps
        origins = {
            event_id: Node(event_id=event_id, stop_id=stop_id)
            for stop_id, event_id in origin_map.items()
        }
        destinations = {
            event_id: Node(event_id=event_id, stop_id=stop_id)
            for stop_id, event_id in destination_map.items()
        }
        return origins, destinations

    @property
    def events_all(self) -> dict[int, Node]:
        origins, destinations = self.events_aux
        return origins | destinations | self.events

    @staticmethod
    def _get_ods_mapped(
        od_node_id_maps: pair[dict[int, int]],
        ods: dict[pair[int], OD],
    ) -> dict[pair[int], OD]:
        origin_map, destination_map = od_node_id_maps
        return {
            (origin_map[u], destination_map[v]): od
            for (u, v), od in ods.items()
        }

    @classmethod
    def from_path(cls, path: str) -> 'Data':
        config = cls.read_config(path)
        return cls(
            config=config,
            activities=cls.read_activities(path, config=config),
            events=cls.read_events(path),
            ods=cls.read_od(path),
        )

    @staticmethod
    def read_config(path: str) -> Config:
        filename = os.path.join(path, Filename.CONFIG.value)
        config = read(filename).set_index('config_key')['value'].to_dict()
        return Config(
            ptn_name=str(config['ptn_name']),
            period_length=int(config['period_length']),
            ean_change_penalty=int(config['ean_change_penalty']),
        )

    @staticmethod
    def read_activities(
        path: str,
        config: Config,
    ) -> dict[pair[int], Activity]:
        filename = os.path.join(path, Filename.ACTIVITIES.value)
        df = read(filename)
        activities = [
            Activity(**row.to_dict(), penalty=config.ean_change_penalty)
            for _, row in df.iterrows()
        ]
        return {
            (activity.from_event, activity.to_event): activity
            for activity in activities
        }

    @staticmethod
    def read_events(path: str) -> dict[int, Event]:
        filename = os.path.join(path, Filename.EVENTS.value)
        df = read(filename)
        events = [
            Event(**row.to_dict()) for _, row in df.iterrows()
        ]
        return {
            event.event_id: event for event in events
        }

    @staticmethod
    def read_od(path: str) -> dict[pair[int], OD]:
        filename = os.path.join(path, Filename.OD.value)
        df = read(filename)
        ods = [
            OD(**row.to_dict()) for _, row in df.iterrows()
        ]
        return {
            (od.origin, od.destination): od
            for od in ods
        }

    def save(self, folder: str):
        path = pathlib.Path(folder)
        path.mkdir(exist_ok=True)
        with open(path / Filename.CONFIG.value, 'w') as file:
            file.write(self.get_config_csv())
        with open(path / Filename.EVENTS.value, 'w') as file:
            file.write(self.get_events_csv())
        with open(path / Filename.ACTIVITIES.value, 'w') as file:
            file.write(self.get_activities_csv())
        with open(path / Filename.OD.value, 'w') as file:
            file.write(self.get_ods_csv())
        if self.solution is None:
            return
        pd.DataFrame(
            data=self.solution.timetable.items(),
            columns=['# event_id', 'time'], dtype=int
        ).to_csv(path / Filename.TIMETABLE.value, sep=';', index=False)
        pd.DataFrame(
            data=[(*uv, *ij) for uv, ij in self.solution.used_edges],
            columns=list('uvij'),
        ).to_csv(path / Filename.PATHS.value, sep=';', index=False)
        pd.DataFrame(
            data=[
                (*ij, weight) for ij, weight in self.solution.weights.items()
            ],
            columns=['i', 'j', 'weight'],
        ).to_csv(path / Filename.WEIGHTS.value, sep=';', index=False)

    def get_config_csv(self) -> str:
        return (
            '# config_key; value\n'
            f'ptn_name;{self.config.ptn_name}\n'
            f'period_length;{self.config.period_length}\n'
            f'ean_change_penalty;{self.config.ean_change_penalty}'
        )

    def get_activities_csv(self) -> str:
        header = '# activity_index;type;from_event;to_event;lower_bound;upper_bound\n'

        def to_row(act: Activity) -> str:
            return f'{act.activity_index};{act.type.value};{act.from_event};{act.to_event};{act.lower_bound};{act.upper_bound}'
        return header + '\n'.join(to_row(activity) for activity in self.activities_constrainable.values())

    def get_events_csv(self) -> str:
        header = '# event_id;type;stop_id;line_id;line_direction;line_freq_repetition\n'

        def to_row(event: Event) -> str:
            return f'{event.event_id};{event.type.value};{event.stop_id};{event.line_id};{event.line_direction.value};{event.line_freq_repetition}'

        return header + '\n'.join(to_row(event) for event in self.events.values())

    def get_ods_csv(self) -> str:
        header = '# origin;destination;customers\n'

        def to_row(od: OD) -> str:
            return f'{od.origin};{od.destination};{od.customers}'

        return header + '\n'.join(to_row(od) for od in self.ods.values())
