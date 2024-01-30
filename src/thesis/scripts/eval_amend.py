import logging
from typing import Callable, Mapping, Any
from functools import partial
from io import StringIO
import subprocess
import pickle

from thesis import add_log_file_handler
from argparse import Namespace, ArgumentParser
import pandas as pd

from thesis.data.wrapper import Data


logger = logging.getLogger(__name__)


def attach_eval_amend(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(
        name='eval-amend', help='Amend the evaluation with dataset instance stats.'
    )
    parser.add_argument(
        '--eval-path', type=str, required=True, help='Evaluation file path'
    )
    parser.add_argument(
        '--amend-path', type=str, required=True, help='Amended file path'
    )
    parser.add_argument(
        '--fake-data-path', type=str, required=False, help='Path for local test data.', default=None
    )
    parser.set_defaults(func=eval_amend)


def get_stats(filename: str):
    instance = Data.from_pickle(filename)
    stop_set = set()
    line_set = set()
    line_freq_set = set()

    for event in instance.events.values():
        stop_set.add(event.stop_id)
        line_set.add(event.line_id)
        line_freq_set.add((event.line_id, event.line_freq_repetition))
    num_stops = len(stop_set)
    num_lines = len(line_set)
    num_lines_freqa = len(line_freq_set)
    num_activities = len(instance.activities_constrainable)
    num_events = len(instance.events)
    num_od_pairs = len(instance.ods)
    return {
        'stops': num_stops,
        'lines': num_lines,
        'line_reps': num_lines_freqa,
        'activities': num_activities,
        'events': num_events,
        'od_pairs': num_od_pairs,
    }



def modify_for_local_test(df: pd.DataFrame, fake_data_path: str):
    from pathlib import Path
    filenames = list(Path(fake_data_path).glob('**/*.pkl.gz'))
    assert len(filenames) >= 5
    df = df.head(5)
    df.loc[:, 'filename'] = filenames[:5]
    return df


def eval_amend(args: Namespace):
    add_log_file_handler('eval_amend.log')

    eval_filename = args.eval_path
    eval_amend_filename = args.amend_path

    logger.info('Loading results')
    with open(eval_filename, 'rb') as file:
        result = pickle.load(file)
    
    if args.fake_data_path is not None:
        logger.info('Testing with fake data')
        result = modify_for_local_test(result, args.fake_data_path)
    
    assert isinstance(result, pd.DataFrame)
    assert 'filename' in result.columns

    logger.info('calculating statistics')
    stats = result['filename'].apply(get_stats)
    stats = pd.DataFrame.from_records(stats)

    amended_results = result.join(stats)
    logger.info('Dumping to a pickle file')
    with open(eval_amend_filename, 'wb') as file:
        pickle.dump(amended_results, file)
    logger.info('Done')
