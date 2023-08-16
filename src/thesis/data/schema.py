from attrs import define, field
from dataclasses import dataclass

from enum import Enum


def convert_int(value) -> int:
    if isinstance(value, int):
        return value
    else:
        return int(value)


class ParseEnum(Enum):
    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError
        return cls(value.strip(' "'))


class ActivityType(ParseEnum):
    WAIT = 'wait'
    DRIVE = 'drive'
    CHANGE = 'change'
    SYNC = 'sync'
    HEADWAY = 'headway'


class EventType(ParseEnum):
    DEPARTURE = 'departure'
    ARRIVAL = 'arrival'


class Direction(ParseEnum):
    FORWARDS = '>'
    BACKWARDS = '<'


@define
class Edge:
    activity_index: int = field(converter=convert_int)
    from_event: int = field(converter=convert_int)
    to_event: int = field(converter=convert_int)
    lower_bound: int = field(default=0, init=False)
    upper_bound: int = field(default=0, init=False)
    penalty: int = field(default=0, init=False)


@define
class Activity(Edge):
    type: ActivityType = field(converter=ActivityType.parse)
    lower_bound: int = field(converter=convert_int)
    upper_bound: int = field(converter=convert_int)
    penalty: int = field(converter=convert_int)

    def __attrs_post_init__(self):
        if self.type != ActivityType.CHANGE:
            self.penalty = 0


@define
class Node:
    event_id: int = field(converter=convert_int)
    stop_id: int = field(converter=convert_int)


@define
class Event(Node):
    type: EventType = field(converter=EventType.parse)
    line_id: int = field(converter=convert_int)
    line_direction: Direction = field(converter=Direction.parse)
    line_freq_repetition: int = field(converter=convert_int)


@define
class OD:
    origin: int = field(converter=convert_int)
    destination: int = field(converter=convert_int)
    customers: int = field(converter=convert_int)


@dataclass
class Config:
    ptn_name: str
    period_length: int
    ean_change_penalty: int
