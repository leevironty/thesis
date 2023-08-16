from typing import Iterable
import random

import networkx as nx

from thesis.data.wrapper import Data, _Counter
from thesis.data.schema import (
    OD,
    ActivityType,
    Activity,
    EventType,
    Event,
    Config,
    Direction,
)


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
            for key, od in rng.sample(list(data.ods.items()), k=od_size)
        }
        new_data = Data(
            config=data.config,
            activities={
                key: value
                for key, value in data.activities.items()
                if rng.random() > activity_drop_prob
                or value.type != ActivityType.CHANGE
            },
            events=data.events,
            ods=new_ods,
        )
        yield new_data


def get_all_paths() -> list[list[tuple[int, int]]]:
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 5),
        (5, 6),
        (5, 7),
        (2, 8),
    ]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nodes = list(range(1, 9))
    all_paths: list[list[tuple[int, int]]] = []

    while nodes:
        source = nodes.pop(0)
        all_paths += list(nx.all_simple_edge_paths(graph, source, nodes))
    return all_paths


def toy_variations(n: int, seed: int) -> Iterable[Data]:
    rng = random.Random()
    rng.seed(seed)
    all_paths = get_all_paths()
    max_attempts = 100

    for sample in range(n):
        got_connected = False
        for _ in range(max_attempts):
            num_lines = rng.randint(2, 4)
            paths = rng.sample(all_paths, k=num_lines)
            g = nx.Graph()
            for line in paths:
                g.add_edges_from(line)
            if nx.is_connected(g):
                got_connected = True
                break
        if not got_connected:
            raise RuntimeError(
                f'Did not find a connected line plan in {max_attempts} attempts.'
            )
        pairs = [(u, v) for u in g.nodes for v in g.nodes if u != v]
        if len(pairs) > 40:
            k = rng.randint(30, 40)
            pairs = rng.sample(pairs, k=k)
        ods = [
            OD(origin=u, destination=v, customers=rng.randint(0, 20))
            for (u, v) in pairs
        ]
        event_counter = _Counter(0)
        activity_counter = _Counter(0)
        events: list[Event] = []
        activities: list[Activity] = []
        for line_id, line in enumerate(paths):
            # emphasis on lower line freqs if we already have many lines
            freq = int((rng.random() ** (num_lines - 1)) // 0.25) + 1
            # line_count = rng.randint(1, 4)
            for rep in range(1, freq + 1):
                prev_arr_f: Event | None = None
                prev_dep_b: Event | None = None
                for i, j in line:
                    dep_f = Event(
                        event_id=event_counter.next(),
                        stop_id=i,
                        type=EventType.DEPARTURE,
                        line_id=line_id + 1,
                        line_direction=Direction.FORWARDS,
                        line_freq_repetition=rep,
                    )
                    arr_f = Event(
                        event_id=event_counter.next(),
                        stop_id=j,
                        type=EventType.ARRIVAL,
                        line_id=line_id + 1,
                        line_direction=Direction.FORWARDS,
                        line_freq_repetition=rep,
                    )
                    dep_b = Event(
                        event_id=event_counter.next(),
                        stop_id=j,
                        type=EventType.DEPARTURE,
                        line_id=line_id + 1,
                        line_direction=Direction.BACKWARDS,
                        line_freq_repetition=rep,
                    )
                    arr_b = Event(
                        event_id=event_counter.next(),
                        stop_id=i,
                        type=EventType.ARRIVAL,
                        line_id=line_id + 1,
                        line_direction=Direction.BACKWARDS,
                        line_freq_repetition=rep,
                    )
                    events += [dep_f, arr_f, dep_b, arr_b]
                    lb = rng.randint(1, 15)
                    ub = lb + rng.randint(0, 5)
                    drive_f = Activity(
                        type=ActivityType.DRIVE,
                        activity_index=activity_counter.next(),
                        from_event=dep_f.event_id,
                        to_event=arr_f.event_id,
                        lower_bound=lb,
                        upper_bound=ub,
                        penalty=0,
                    )
                    drive_b = Activity(
                        type=ActivityType.DRIVE,
                        activity_index=activity_counter.next(),
                        from_event=dep_b.event_id,
                        to_event=arr_b.event_id,
                        lower_bound=lb,
                        upper_bound=ub,
                        penalty=0,
                    )
                    activities += [drive_f, drive_b]
                    if prev_arr_f is not None and prev_dep_b is not None:
                        lb = rng.randint(1, 3)
                        ub = lb + rng.randint(0, 2)
                        change_f = Activity(
                            type=ActivityType.CHANGE,
                            activity_index=activity_counter.next(),
                            from_event=prev_arr_f.event_id,
                            to_event=dep_f.event_id,
                            lower_bound=lb,
                            upper_bound=ub,
                            penalty=0,
                        )
                        change_b = Activity(
                            type=ActivityType.CHANGE,
                            activity_index=activity_counter.next(),
                            from_event=arr_b.event_id,
                            to_event=prev_dep_b.event_id,
                            lower_bound=lb,
                            upper_bound=ub,
                            penalty=0,
                        )
                        activities += [change_f, change_b]
                    prev_arr_f = arr_f
                    prev_dep_b = dep_b
        # add transfers
        for stop_id in g.nodes:
            arrs = [
                event
                for event in events
                if event.stop_id == stop_id and event.type == EventType.ARRIVAL
            ]
            deps = [
                event
                for event in events
                if event.stop_id == stop_id and event.type == EventType.DEPARTURE
            ]
            for arr in arrs:
                for dep in deps:
                    if arr.line_id == dep.line_id:
                        continue
                    lb = rng.randint(1, 5)
                    ub = lb + 59
                    transfer = Activity(
                        type=ActivityType.CHANGE,
                        activity_index=activity_counter.next(),
                        from_event=arr.event_id,
                        to_event=dep.event_id,
                        lower_bound=lb,
                        upper_bound=ub,
                        penalty=0,
                    )
                    activities.append(transfer)
        ods_map = {(od.origin, od.destination): od for od in ods}
        events_map = {event.event_id: event for event in events}
        activities_map = {
            (activity.from_event, activity.to_event): activity
            for activity in activities
        }
        config = Config(
            ptn_name=f'generated_{sample:06}',
            period_length=60,
            ean_change_penalty=rng.randint(0, 5),
        )
        data = Data(
            config=config, activities=activities_map, events=events_map, ods=ods_map
        )
        # test_g = nx.DiGraph()
        # test_g.add_edges_from(data.activities_routable)
        # origins, destinations = data.events_aux
        # for origin in origins:
        #     for destination in destinations:
        #         origin_stop = data.events_aux[0][origin].stop_id
        #         destination_stop = data.events_aux[1][destination].stop_id
        #         if origin_stop == destination_stop:
        #             continue
        #         if not nx.has_path(test_g, origin, destination):
        #             raise RuntimeError('No path!')
        yield data
