from typing import Iterable
from thesis.data.wrapper import Data
import random


def variations(data: Data, n: int, od_share: float) -> Iterable[Data]:
    rng = random.Random()
    rng.seed(847120395)
    od_size = int(len(data.ods) * od_share)

    for _ in range(n):
        new_data = Data(
            config=data.config,
            activities=data.activities,
            events=data.events,
            ods=dict(rng.sample(list(data.ods.items()), k=od_size))
        )
        yield new_data
