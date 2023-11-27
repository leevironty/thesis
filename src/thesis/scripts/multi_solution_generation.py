from thesis.scripts.multi_solution_check import MultiSolutionTimPass


from argparse import Namespace, ArgumentParser
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
from thesis.models.mip.pesp import PESP
from thesis.models.mip.preprocessor import preprocess
from thesis.models.mip.solver import get_solver
from pulp import LpStatusOptimal


logger = logging.getLogger(__name__)


def attach_multi_solution_generation(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(name='multi-solution-generation', help='Generate data with multiple solutions')
    parser.add_argument('--out', type=str, required=True, help='logpath')
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--time-limit', type=int, default=20)
    parser.add_argument('--alt-solutions', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--share', type=float, required=True)
    parser.set_defaults(func=generate_multi_solution_data)



def generate_multi_solution_data(args: Namespace):
    output_folder = pathlib.Path(args.out) / args.job_id / args.job_array_id
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / 'logs.log'
    add_log_file_handler(log_path.as_posix())

    logger.info(f'Starting the data generation process with arguments: {args}')
    logger.info('Generating variations and solving the problems.')

    start = time.time()
    iterations = toy_variations(n=args.count, seed=args.seed + 7192)
    solver = get_solver(
        threads=args.threads, time_limit=args.time_limit, output=False,
    )
    unique_count = 0
    attempt_count = 0
    for problem_number, variation in enumerate(iterations):
        logger.info(f'Generating solutions for problem {problem_number}')
        variation.preprocessed_flows = preprocess(variation)
        model = MultiSolutionTimPass(variation, epsilon=0.0001, solver=solver)
        if model.original_timpass.model.sol_status != LpStatusOptimal:
            logger.info(f'Skipping problem {problem_number}, did not find an optimal solution.')
            continue
        uniques = []
        for attempt_number in range(args.alt_solutions):
            logger.info(f'Generating attempt {attempt_number}')
            try:
                attempt_count += 1
                preferences = model.get_solution()
                solution = model.original_timpass.get_solution()

            except RuntimeError:
                logger.info('Did not find an optimal solution in the given time limit')
                continue
            if solution.weights not in uniques:
                unique_count += 1
                uniques.append(solution.weights)
                variation.preferences = preferences
                variation.solution = solution
                filename = f'solution_{problem_number:04}_{len(uniques):02}.pkl.gz'
                logger.info(f'Found unique solution, saving as {filename}')

                # sanity check for weird objective calculation issues
                timpass_obj = model.original_timpass.objective.value()
                pesp = PESP(variation, solution.weights, solver)
                pesp.solve()
                pesp_obj = pesp.objective.value()
                if abs(timpass_obj - pesp_obj) > 0.01:
                    print(f'pesp sanity check not passed: {timpass_obj=}, {pesp_obj=}, {output_folder / filename}')
                solution_obj_value = 0
                for uv, od in variation.ods_mapped.items():
                    for ij, act in variation.activities.items():
                        if (uv, ij) in solution.used_edges:
                            solution_obj_value += od.customers * (solution.edge_durations[ij] + act.penalty)
                if abs(solution_obj_value - timpass_obj) > 0.01:
                    logger.warning(f'solution sanity check not passed: {timpass_obj=}, {solution_obj_value=}, {output_folder / filename}')
                # sanity checks done

                with gzip.open(output_folder / filename, 'wb') as file:
                    pickle.dump(variation, file)


        # model.solve()
        # if model.model.sol_status != pulp.LpSolutionOptimal:
        #     logger.warning(
        #         f'Solution for problem {i} is not optimal! '
        #         f'Status: {model.model.sol_status}'
        #     )
        #     path = output_folder / f'non-optimal_{i:05}.pkl.gz'
        # else:
        #     variation.solution = model.get_solution()
        #     path = output_folder / f'solution_{i:05}.pkl.gz'
        # logger.info(f'Solved problem {i}')
        # with gzip.open(path, 'wb') as file:
        #     pickle.dump(variation, file)
    end = time.time()
    logger.info(f'Found {unique_count} / {attempt_count} unique solutions in {end - start :.2f} seconds.')
