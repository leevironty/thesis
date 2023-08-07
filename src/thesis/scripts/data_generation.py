from argparse import Namespace
import time
import logging
import pathlib

from thesis.data.wrapper import Data
from thesis.data.generator import variations
from models.timpass import TimPass
from models.preprocessor import preprocess


logger = logging.getLogger(__name__)


def generate_data(args: Namespace):
    logger.info(f'Starting the data generation process with arguments: {args}')
    logger.info('Reading the base dataset')
    dataset = Data.from_path(args.dataset)
    pathlib.Path(args.out).mkdir(exist_ok=True)
    logger.info('Generating variations and solving the problems.')
    start = time.time()
    for i, variation in enumerate(variations(
        data=dataset,
        n=args.count,
        od_share=args.od_share,
        seed=args.seed,
    )):
        if args.preprocess:
            variation.preprocessed_flows = preprocess(variation)
        model = TimPass(variation)
        model.solve()
        variation.solution = model.get_solution()
        variation.save(f'{args.out}/{i}')
    end = time.time()
    logger.info(f'Solved {args.count} problems in {end - start :.2f} seconds.')
