# from dataclasses import dataclass
from attrs import define, field
from functools import cached_property
import os
from enum import Enum
import pandas as pd
# from pandera.typing import DataFrame

from thesis.data.schema import Activity, Config, OD, Event


# TODO: python 3.11 support StrEnum
class Filename(Enum):
    ACTIVITIES = "Activities.csv"
    CONFIG = "Config.csv"
    EVENTS = "Events.csv"
    OD = "OD.csv"


def read(path: str) -> pd.DataFrame:
    with open(path) as file:
        file.seek(2)  # Drop comment hashtags
        return pd.read_csv(file, sep=';').rename(columns=str.strip)


@define
class Solution:
    timetable: dict[int, int]
    weights: dict[int, int]


@define
class Data:
    config: Config
    # activities: list[Activity]
    activities: dict[tuple[int, int], Activity]
    # events: list[Event]
    events: dict[int, Event]
    # od: list[OD]
    od: dict[tuple[int, int], OD]
    solution: Solution | None = field(default=None)

    @cached_property
    def activities_aux(self) -> dict[tuple[int, int], Activity]:
        ...

    @property
    def activities_all(self) -> dict[tuple[int, int], Activity]:
        ...

    @cached_property
    def events_aux(self) -> dict[int, Event]:
        ...

    @property
    def events_all(self) -> dict[int, Event]:
        ...

    @classmethod
    def from_path(cls, path: str) -> 'Data':
        return cls(
            config=cls.read_config(path),
            activities=cls.read_activities(path),
            events=cls.read_events(path),
            od=cls.read_od(path),
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
    def read_activities(path: str) -> dict[tuple[int, int], Activity]:
        filename = os.path.join(path, Filename.ACTIVITIES.value)
        df = read(filename)
        activities = [
            Activity(**row.to_dict()) for _, row in df.iterrows()
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
    def read_od(path: str) -> dict[tuple[int, int], OD]:
        filename = os.path.join(path, Filename.OD.value)
        df = read(filename)
        ods = [
            OD(**row.to_dict()) for _, row in df.iterrows()
        ]
        return {
            (od.origin, od.destination): od
            for od in ods
        }

    @staticmethod
    def generate_aux_activities(partial_data: 'Data') -> list[Activity]:
        ...
