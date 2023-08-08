from typing import Iterable
from thesis.data.wrapper import Data
from thesis.data.schema import OD
import random


def variations(
    data: Data,
    n: int,
    od_share: float,
    activity_drop_prob: float,
    seed: int,
) -> Iterable[Data]:
    rng = random.Random()
    rng.seed(seed)
    od_size = int(len(data.ods) * od_share)

    for _ in range(n):
        new_ods = {
            key: OD(
                origin=od.origin,
                destination=od.destination,
                customers=int(od.customers * (rng.random() * 4.8 + 0.2)),
            )
            for key, od in rng.sample(list(data.ods.items()), k=od_size)}
        new_data = Data(
            config=data.config,
            activities={
                key: value for key, value in data.activities.items()
                if rng.random() > activity_drop_prob
            },
            events=data.events,
            ods=new_ods,
        )
        yield new_data
