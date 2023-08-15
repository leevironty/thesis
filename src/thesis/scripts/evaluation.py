from argparse import Namespace
import logging
import pathlib

from thesis.data.wrapper import Data
from thesis import add_log_file_handler
from thesis.evaluation.model import Baseline
from thesis.models.mip.pesp import PESP
from thesis.models.mip.solver import get_solver


logger = logging.getLogger(__name__)


def run_evaluation(args: Namespace):
    output_folder = pathlib.Path(args.out) / args.job_id / args.job_array_id
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / 'logs.log'
    add_log_file_handler(log_path.as_posix())

    logger.info(f'Starting the evaluation process with arguments: {args}')
    logger.info('Reading the evaluation dataset')
    dataset = Data.from_path(args.dataset)
    solver_log_path = output_folder / 'pesp_solver.log'
    solver = get_solver(
        log_path=solver_log_path.as_posix(),
        threads=args.threads,
        time_limit=args.time_limit,
    )

    if args.baseline:
        logger.info('Evaluating baseline solution')
        baseline = Baseline(dataset)
        weights = baseline.get_weights()
        logger.info('Got weights, running the PESP model')
        model = PESP(dataset, weights=weights, solver=solver)
        model.solve()
        value = model.objective.value()
        logger.info(f'Got baseline objective value {value}')
