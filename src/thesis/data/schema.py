from attrs import define, field
from dataclasses import dataclass

from enum import Enum
# import pandera as pa
# from pandera.typing import Series


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


class EventType(ParseEnum):
    DEPARTURE = 'departure'
    ARRIVAL = 'arrival'


class Direction(ParseEnum):
    FORWARDS = '>'
    BACKWARDS = '<'


@define
class Activity:
    activity_index: int = field(converter=int)
    type: ActivityType = field(converter=ActivityType.parse)
    from_event: int = field(converter=int)
    to_event: int = field(converter=int)
    lower_bound: int = field(converter=int)
    upper_bound: int = field(converter=int)


@define
class Event:
    event_id: int = field(converter=int)
    type: EventType = field(converter=EventType.parse)
    stop_id: int = field(converter=int)
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
