from argparse import Namespace
import time
import logging
import pathlib
import pickle
import gzip

from thesis.data.wrapper import Data
from thesis import add_log_file_handler
from thesis.data.generator import toy_variations

# from thesis.models.mip.timpass import TimPass
from thesis.models.mip.cycle_basis import TimPassCycle
from thesis.models.mip.preprocessor import preprocess
from thesis.models.mip.solver import get_solver
import pulp


logger = logging.getLogger(__name__)


def generate_data(args: Namespace):
    output_folder = pathlib.Path(args.out) / args.job_id / args.job_array_id
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / 'logs.log'
    add_log_file_handler(log_path.as_posix())

    logger.info(f'Starting the data generation process with arguments: {args}')
    logger.info('Reading the base dataset')
    logger.info('Generating variations and solving the problems.')

    # Model = TimPassCycle if args.cycle else TimPass
    Model = TimPassCycle
    start = time.time()
    if args.variations:
        # iterations = variations(
        #     data=dataset,
        #     n=args.count,
        #     od_share=args.od_share,
        #     seed=args.seed,
        #     activity_drop_prob=args.activity_drop_prob,
        # )
        iterations = toy_variations(n=args.count, seed=args.seed)
    else:
        dataset = Data.from_path(args.dataset)
        iterations = [dataset]
    for i, variation in enumerate(iterations):
        log_path = output_folder / f'solution_{i:05}_log.log'
        solver = get_solver(
            log_path=log_path.as_posix(), threads=args.threads, time_limit=args.time_limit
        )
        if args.preprocess:
            variation.preprocessed_flows = preprocess(variation)
        model = Model(variation, solver)
        model.solve()
        if model.model.sol_status != pulp.LpSolutionOptimal:
            logger.warning(
                f'Solution for problem {i} is not optimal! '
                f'Status: {model.model.sol_status}'
            )
            path = output_folder / f'non-optimal_{i:05}.pkl.gz'
        else:
            variation.solution = model.get_solution()
            path = output_folder / f'solution_{i:05}.pkl.gz'
        logger.info(f'Solved problem {i}')
        with gzip.open(path, 'wb') as file:
            pickle.dump(variation, file)
    end = time.time()
    logger.info(f'Solved {args.count} problems in {end - start :.2f} seconds.')
