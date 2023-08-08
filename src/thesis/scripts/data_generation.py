from argparse import Namespace
import time
import logging
import pathlib
import pickle
import gzip

from thesis.data.wrapper import Data
from thesis import add_log_file_handler
from thesis.data.generator import variations
from models.timpass import TimPass
from models.cycle_basis import TimPassCycle
from models.preprocessor import preprocess
from models.solver import get_solver
import pulp


logger = logging.getLogger(__name__)


def generate_data(args: Namespace):
    output_folder = pathlib.Path(args.out) / args.job_id / args.job_array_id
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / 'logs.log'
    add_log_file_handler(log_path.as_posix())

    logger.info(f'Starting the data generation process with arguments: {args}')
    logger.info('Reading the base dataset')
    dataset = Data.from_path(args.dataset)
    logger.info('Generating variations and solving the problems.')

    # Model = TimPassCycle if args.cycle else TimPass
    Model = TimPassCycle
    start = time.time()
    for i, variation in enumerate(variations(
        data=dataset,
        n=args.count,
        od_share=args.od_share,
        seed=args.seed,
        activity_drop_prob=args.activity_drop_prob,
    )):
        log_path = output_folder / f'solution_{i:05}_log.log'
        solver = get_solver(log_path.as_posix(), threads=args.threads, time_limit=args.time_limit)
        if args.preprocess:
            variation.preprocessed_flows = preprocess(variation)
        model = Model(variation, solver)
        model.solve()
        if model.model.sol_status != pulp.LpSolutionOptimal:
            logger.warning(
                f'Solution for problem {i} is not optimal! '
                f'Status: {model.model.sol_status}'
            )
            path = output_folder / 'non-optimal_{i:05}.pkl.gz'
        else:
            variation.solution = model.get_solution()
            path = output_folder / f'solution_{i:05}.pkl.gz'
        with gzip.open(path, 'wb') as file:
            pickle.dump(variation, file)
    end = time.time()
    logger.info(f'Solved {args.count} problems in {end - start :.2f} seconds.')
