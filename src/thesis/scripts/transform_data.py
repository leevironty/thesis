from argparse import Namespace, ArgumentParser
from thesis.data.wrapper import Data
import torch
import tqdm

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


def run_transform(args: Namespace):
    logger.info('Searching for data files')
    dataset_path = Path(args.dataset)
    # put to list to make tqdm aware of the total number of files
    solutions = list(dataset_path.glob('**/solution_*.pkl.gz'))
    logger.info('Converting')
    for filename in tqdm.tqdm(solutions):
        data = Data.from_pickle(filename.as_posix())
        hetero_data = transform(data)
        hetero_path = filename.with_suffix('').with_suffix('.pt')
        torch.save(hetero_data, hetero_path.as_posix())
    logger.info('All transformations done!')
