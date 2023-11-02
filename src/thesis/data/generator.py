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
        all_paths += list(
            path for path in nx.all_simple_edge_paths(graph, source, nodes)
            if len(path) >= 2
        )
    return all_paths


def toy_variations(n: int, seed: int) -> Iterable[Data]:
    rng = random.Random()
    rng.seed(seed)
    all_paths = get_all_paths()
    max_attempts = 100
    penalty = rng.randint(1, 5)

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
            OD(origin=u, destination=v, customers=rng.randint(1, 20))
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
            bounds_drive: dict[tuple[int, int], tuple[int, int]] = {}
            bounds_wait: dict[tuple[int, int], tuple[int, int]] = {}
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
                    if (i, j) not in bounds_drive:
                        # Duration must be the same for each repetition
                        lb = rng.randint(1, 15)
                        ub = lb + rng.randint(0, 5)
                        bounds_drive[(i, j)] = (lb, ub)
                    else:
                        lb, ub = bounds_drive[(i, j)]
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
                        # same stop should have the same change times for the given line between different repetitions
                        if (i, j) not in bounds_wait:
                            lb = rng.randint(1, 3)
                            ub = lb + rng.randint(0, 2)
                            bounds_wait[(i, j)] = (lb, ub)
                        else:
                            lb, ub = bounds_wait[(i, j)]
                        change_f = Activity(
                            type=ActivityType.WAIT,
                            activity_index=activity_counter.next(),
                            from_event=prev_arr_f.event_id,
                            to_event=dep_f.event_id,
                            lower_bound=lb,
                            upper_bound=ub,
                            penalty=0,
                        )
                        change_b = Activity(
                            type=ActivityType.WAIT,
                            activity_index=activity_counter.next(),
                            from_event=arr_b.event_id,
                            to_event=prev_dep_b.event_id,
                            lower_bound=lb,
                            upper_bound=ub,
                            penalty=0,
                        )
                        activities += [change_f, change_b]

                    if rep > 1:
                        interval = int(60 / freq)
                        event_id_offset = 4 * len(line)
                        for latter_sync_event in [arr_f, arr_b, dep_f, dep_b]:
                            sync = Activity(
                                type=ActivityType.SYNC,
                                activity_index=activity_counter.next(),
                                from_event=latter_sync_event.event_id - event_id_offset,
                                to_event=latter_sync_event.event_id,
                                lower_bound=interval,
                                upper_bound=interval,
                                penalty=0,
                            )
                            activities.append(sync)

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
            lb = rng.randint(1, 5)
            ub = lb + 59
            for arr in arrs:
                for dep in deps:
                    if arr.line_id == dep.line_id:
                        continue
                    transfer = Activity(
                        type=ActivityType.CHANGE,
                        activity_index=activity_counter.next(),
                        from_event=arr.event_id,
                        to_event=dep.event_id,
                        lower_bound=lb,
                        upper_bound=ub,
                        penalty=penalty,
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
            ean_change_penalty=penalty,
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
