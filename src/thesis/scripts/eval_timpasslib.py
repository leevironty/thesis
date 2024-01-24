import logging
from typing import Callable, Mapping
from functools import partial
from io import StringIO
import subprocess

from thesis import add_log_file_handler
from argparse import Namespace, ArgumentParser
import pandas as pd
import numpy as np
import shutil
import networkx as nx

from thesis.models.gnn.hgt import Predictor
from thesis.data.wrapper import Data, pair, ActivityType
from thesis.models.mip.preprocessor import preprocess
from thesis.scripts.evaluation import get_gnn_weights, get_trivial_weights, pair


logger = logging.getLogger(__name__)


def attach_eval_timpasslib(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(
        name='eval-timpasslib', help='Prepare LinTim datasets'
    )
    parser.add_argument(
        '--time-limit', type=int, required=True, help='Solver time limit in minutes'
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.set_defaults(func=run_eval_timpasslib)


DATASET_NAMES = [
    'toy_2',
    'grid',
    # 'regional',
    # 'Erding_NDP_S020',
    # 'R2L1',  # very slow to even load??
]


def get_activities_df(data: Data, weights: dict[pair, float]) -> pd.DataFrame:
    columns = '# activity_index; type; from_event; to_event; lower_bound; upper_bound; passengers'.split(
        '; '
    )
    df = pd.DataFrame.from_records(
        data=[
            (
                act.activity_index,
                f'"{act.type.value}"',
                act.from_event,
                act.to_event,
                act.lower_bound,
                act.upper_bound,
                weights.get(ij, 0),
            )
            for ij, act in data.activities_constrainable.items()
        ],
        columns=columns,
    )
    return df


def get_stops_df(events_df: pd.DataFrame) -> pd.DataFrame:
    stop_ids = events_df['stop-id'].unique()
    long_names = [f'stop-{i}' for i in stop_ids]

    out = pd.DataFrame(
        data={
            '# stop-id': stop_ids,
            'short-name': stop_ids,
            'long-name': long_names,
            'x-coordinate': np.random.randint(0, 100, size=len(long_names)),  # only for visuals
            'y-coordinate': np.random.randint(0, 100, size=len(long_names)),
        }
    )
    return out.sort_values(by='# stop-id')






def get_dfs(
    data: Data, weights: dict[pair, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activities_df = get_activities_df(data, weights)
    inbound = activities_df.groupby('to_event')['passengers'].sum()
    outbound = activities_df.groupby('from_event')['passengers'].sum()
    passengers = inbound.subtract(outbound, fill_value=0)

    columns = '# event_id; type; stop-id; line-id; passengers; line-direction; line-freq-repetition'.split(
        '; '
    )
    df = pd.DataFrame.from_records(
        data=[
            (
                event.event_id,
                f'"{event.type.value}"',
                event.stop_id,
                event.line_id,
                passengers[event.event_id],
                event.line_direction.value,
                event.line_freq_repetition,
            )
            for _, event in data.events.items()
        ],
        columns=columns,
    )
    return df, activities_df


def get_activities_csv(data: Data, weights: dict[pair, float]):
    yield '# activity_index; type; from_event; to_event; lower_bound; upper_bound; passengers\n'

    for key, activity in data.activities_constrainable.items():
        if key not in weights:
            assert activity.type == ActivityType.SYNC
            weight = 0
        else:
            weight = weights[key]
        yield f'{activity.activity_index}; "{activity.type}"; {activity.from_event}; {activity.to_event}; {activity.lower_bound}; {activity.upper_bound}; {weight}\n'


def get_events_csv(data: Data):
    yield '# event_id; type; stop-id; line-id; passengers; line-direction; line-freq-repetition\n'
    for key, event in data.events.items():
        yield f'{event.event_id}; {event.type.value}; {event.stop_id}; {event.line_id};'


def timetable_to_obj_value(data: Data, timetable: Mapping[int, int]) -> int:
    durations: dict[pair, int] = {}
    for (i, j), act in data.activities_constrainable.items():
        durations[(i, j)] = (
            act.lower_bound
            + (timetable[j] - timetable[i] - act.lower_bound)
            % data.config.period_length
        )

    graph = nx.DiGraph()
    graph.add_edges_from(data.activities_routable)

    def weight(u: int, v: int, _) -> int:
        return durations.get((u, v), 0) + data.activities_routable[(u, v)].penalty

    # check for preprocessed routes being used in the evaluation
    assert data.preprocessed_flows is not None
    preprocess = data.preprocessed_flows

    # norm_weights: dict[pair, float] = {}
    objective = 0
    for (u, v), od in data.ods_mapped.items():
        shortest_path = nx.shortest_path(graph, u, v, weight=weight)
        for edge in zip(shortest_path[:-1], shortest_path[1:]):
            if edge in preprocess[(u, v)]:
                raise RuntimeError(
                    'Something is not correct, tried to use a preprocessed edge'
                )
        length: int = nx.shortest_path_length(graph, u, v, weight=weight)
        objective += length * od.customers
    return objective


def get_data(key: str):
    logger.info(f'loading {key}')
    data = Data.from_path(f'timpasslib/{key}')
    data.assign_random_preferences()
    data.preprocessed_flows = preprocess(data)
    logger.info(f'loaded {key}')
    return data


def to_csv_lintim_format(df: pd.DataFrame, filename: str):
    txt = StringIO()
    df.to_csv(txt, sep=';', index=False)
    txt = txt.getvalue().replace(';', '; ').replace('"""', '"')  # ???
    with open(filename, 'w') as file:
        file.write(txt)


def run_eval_timpasslib(args: Namespace):
    add_log_file_handler('eval_timpasslib_prepare_lintim_datasets.log')

    logging.info('loading gnn from checkpoint')
    gnn = Predictor.load_from_checkpoint(args.checkpoint, map_location='cpu')

    logging.info('preparing methods')
    methods: dict[str, Callable[[Data], dict[pair, float]]] = {
        'gnn': partial(get_gnn_weights, predictor=gnn),
        'sp': get_trivial_weights,
    }

    logging.info('loading datasets')
    datasets = {key: get_data(key) for key in DATASET_NAMES}

    DATASET_BASE = '../lintim/datasets'

    for method_name, method in methods.items():
        for dataset_name, dataset in datasets.items():
            logger.info(
                f'Calculating weights with method {method_name} on dataset {dataset_name}'
            )
            weights = method(dataset)
            logger.info('Got weights!')
            target_dataset_path = (
                f'{DATASET_BASE}/evaluation_{dataset_name}_{method_name}'
            )
            logger.info('Preparing lintim dataset')
            shutil.copytree(
                f'{DATASET_BASE}/template', target_dataset_path, dirs_exist_ok=True
            )
            logger.info('Copied the template')

            events, activities = get_dfs(dataset, weights)
            stops = get_stops_df(events)
            events_filename = f'{target_dataset_path}/timetabling/Events-periodic.giv'
            activities_filename = f'{target_dataset_path}/timetabling/Activities-periodic.giv'
            stops_filename = f'{target_dataset_path}/basis/Stop.giv'
            to_csv_lintim_format(events, events_filename)
            to_csv_lintim_format(activities, activities_filename)
            to_csv_lintim_format(stops, stops_filename)
            # events_txt = StringIO()
            # activities_txt = StringIO()
            # stops_txt = StringIO()
            # events.to_csv(events_txt, sep=';', index=False)
            # activities.to_csv(activities_txt, sep=';', index=False)
            # activities.to_csv(activities_txt, sep=';', index=False)
            # events_txt = (
            #     events_txt.getvalue().replace(';', '; ').replace('"""', '"')
            # )  # ???
            # activities_txt = (
            #     activities_txt.getvalue().replace(';', '; ').replace('"""', '"')
            # )  # ???

            # with open(
            #     f'{target_dataset_path}/timetabling/Events-periodic.giv', 'w'
            # ) as file:
            #     file.write(events_txt)

            # with open(
            #     f'{target_dataset_path}/timetabling/Activities-periodic.giv', 'w'
            # ) as file:
            #     file.write(activities_txt)

            with open(f'{target_dataset_path}/basis/Config.cnf', 'r+') as file:
                lines = file.readlines()
                file.seek(0)
                adt = f'tim_pass_use_cycle_base; true\ntim_pass_use_preprocessing; true\ntim_model; cb_ip\ntim_cp_time_limit; {args.time_limit}\ntim_timelimit; {args.time_limit}\n'
                file.writelines(lines[:36] + [adt] + lines[36:])
            logger.info('Wrote the events and activities')

            # subprocess.run(
            #     f'cd {target_dataset_path} && make tim-timetable',
            #     shell=True,
            #     capture_output=False,
            # )

            # timetable = (
            #     pd.read_csv(
            #         f'{target_dataset_path}/timetabling/Timetable-periodic.tim', sep=';'
            #     )
            #     .rename(str.strip, axis='columns')
            #     .set_index('# event-id')['time']
            # )

            # obj_value = timetable_to_obj_value(dataset, timetable.to_dict())
            # logger.info(
            #     f'Got evaluation result: {method_name=}, {dataset_name=}, {obj_value=}'
            # )

            # logger.info('Dataset prepared')
