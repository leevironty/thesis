from argparse import Namespace, ArgumentParser
from multiprocessing import Pool
from thesis.data.wrapper import Data
import pathlib
import torch
import tqdm
import time

from pathlib import Path

from thesis.models.gnn.data import transform

import logging

logger = logging.getLogger(__name__)


def attach_run_parser(main_subparsers) -> None:
    parser: ArgumentParser = main_subparsers.add_parser(
        name='transform', help='Transform thesis Data to geometric HeteroData'
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.set_defaults(func=run_transform)

def convert(filename: pathlib.Path) -> None:
    max_tries = 3
    num_tries = 0
    success = False
    while not success and num_tries < max_tries:
        num_tries += 1
        try:
            data = Data.from_pickle(filename.as_posix())
            hetero_data = transform(data)
            hetero_path = filename.with_suffix('').with_suffix('.pt')
            torch.save(hetero_data, hetero_path.as_posix())
            success = True
        except Exception as e:
            if num_tries < max_tries:
                logger.warning(f'Got some trouble: {e}, trying again in a second.')
                time.sleep(1)
            else:
                logger.error(f'Did not manage to process: {filename.as_posix()}')
                raise e


def run_transform(args: Namespace):
    logger.info('Searching for data files')
    dataset_path = Path(args.dataset)
    # put to list to make tqdm aware of the total number of files
    solutions = list(dataset_path.glob('**/solution_*.pkl.gz'))
    logger.info('Converting')
    with Pool(processes=args.threads) as pool:
        with tqdm.tqdm(total=len(solutions)) as pbar:
            for _ in pool.imap_unordered(convert, solutions):
                pbar.update()
    logger.info('All transformations done!')
