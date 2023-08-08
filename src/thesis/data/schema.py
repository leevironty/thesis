from attrs import define, field
from dataclasses import dataclass

from enum import Enum


class ParseEnum(Enum):
    @classmethod
    def parse(cls, value):
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
    activity_index: int = field(converter=int)
    from_event: int = field(converter=int)
    to_event: int = field(converter=int)
    lower_bound: int = field(default=0, init=False)
    upper_bound: int = field(default=0, init=False)
    penalty: int = field(default=0, init=False)


@define
class Activity(Edge):
    type: ActivityType = field(converter=ActivityType.parse)
    lower_bound: int = field(converter=int)
    upper_bound: int = field(converter=int)
    penalty: int = field(converter=int)

    def __attrs_post_init__(self):
        if self.type != ActivityType.CHANGE:
            self.penalty = 0


@define
class Node:
    event_id: int = field(converter=int)
    stop_id: int = field(converter=int)


@define
class Event(Node):
    type: EventType = field(converter=EventType.parse)
    line_id: int = field(converter=int)
    line_direction: Direction = field(converter=Direction.parse)
    line_freq_repetition: int = field(converter=int)


@define
class OD:
    origin: int = field(converter=int)
    destination: int = field(converter=int)
    customers: int = field(converter=int)


@dataclass
class Config:
    ptn_name: str
    period_length: int
    ean_change_penalty: int
