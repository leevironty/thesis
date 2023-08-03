from typing import Iterable
from thesis.data.wrapper import Data
from random import Random


def variations(data: Data, n: int, seed: int = 1) -> Iterable[Data]:
    od_prob = 0.1
    rng = Random(seed)
    for _ in range(n):
        new_data = Data(
            config=data.config,
            activities=data.activities,
            events=data.events,
            od=[
                od for od in data.od if rng.random() < od_prob
            ]
        )
        yield new_data
