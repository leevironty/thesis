from argparse import Namespace, ArgumentParser
import logging
import pathlib

from torch_geometric.loader import DataLoader
import tqdm
import torch

from thesis.models.gnn.data import get_dataset


logger = logging.getLogger(__name__)


def attach_batcher_parser(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(name='batch', help='Pre-batch data')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.set_defaults(func=run_batch)


def run_batch(args: Namespace):
    logger.info(f'Started batching with args: {args}')
    logger.info('Getting data loader')
    paths = sorted(list(pathlib.Path(args.dataset).glob('**/solution_*.pt')))
    dataset = get_dataset(paths)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.threads,
    )
    base_path = pathlib.Path(args.out)
    base_path.mkdir(parents=True, exist_ok=True)

    logger.info('Batching')
    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
        filename = base_path / f'solution_batch_{i:04}.pt'
        torch.save(batch, filename.as_posix())

    logger.info('All done!')
