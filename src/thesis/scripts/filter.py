from argparse import Namespace, ArgumentParser
import logging
from pathlib import Path
from multiprocessing import Pool
from re import A
import pandas as pd
import time

from torch_geometric.loader import DataLoader
from thesis.models.mip.cycle_basis import TimPassCycle as TimPass
from thesis.models.mip.pesp import PESP

from thesis.scripts.evaluation import get_trivial_weights, get_optimal_weights
import tqdm
import pickle
# import torch

from thesis.data.wrapper import Data
from thesis.models.mip.solver import get_solver


logger = logging.getLogger(__name__)

# from pulp import (
#     LpProblem,
#     LpVariable,
#     LpMinimize,
#     LpInteger,
#     LpBinary,
#     LpAffineExpression,
#     lpSum,
#     LpSolver,
#     LpStatusOptimal,
# )
import random


def attach_filter_parser(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(name='filter', help='Check for multiple solutions')
    parser.add_argument('--dataset', required=True)
    # parser.add_argument('--share', type=float, required=True)
    parser.set_defaults(func=run_filter)


def run_filter(args: Namespace):
    logger.info(f'Started filter with args: {args}')

    filenames = sorted(list(Path(args.dataset).glob('**/solution_*.pkl.gz')))
    # index = int(len(filenames) * args.share)
    # logger.info(f'Will evaluate {index} instances')
    # filenames = filenames[:index]

    # results = []

    # with Pool(processes=args.threads) as pool:
    #     printouts = 0
    #     with tqdm.tqdm(total=len(filenames)) as pbar:
    #         for res in pool.imap_unordered(calculate, filenames):
    #             results.append(res)
    #             if printouts <= args.threads:
    #                 logger.info(f'Done with {res["file"]}')
    #                 printouts += 1
    #             pbar.update()
    
    # df = pd.DataFrame(results)
    # df.to_pickle('multi-solution-evaluation.pkl')
    # numeric = df.select_dtypes('number')
    # logger.info(f'Got means:\n{numeric.mean()}')
    ratios = []
    ratios_opt = []
    out = []
    for filename in filenames[:100]:

        data = Data.from_pickle(filename.as_posix())

        solver = get_solver(threads=args.threads, time_limit=20, output=True)
        weights = get_trivial_weights(data)
        model_spr = PESP(data=data, weights=weights, solver=solver)
        model_opt = PESP(data=data, weights=get_optimal_weights(data), solver=solver)
        model_spr.solve()
        model_opt.solve()
        durations = model_spr.get_durations(allow_non_optimal=False)
        durations_opt = model_opt.get_durations(False)


        lbs_sum = sum(data.activities[a].lower_bound for a in durations.keys())
        lbs_sum_opt = sum(data.activities[a].lower_bound for a in durations_opt.keys())

        ratio = sum(durations.values()) / lbs_sum
        ratio_opt = sum(durations_opt.values()) / lbs_sum_opt
        print(f'ratio: {ratio:.4f}')
        print(f'ratio_opt: {ratio_opt:.4f}')
        ratios.append(ratio)
        ratios_opt.append(ratio_opt)
        out.append({a: {'duration': durations[a], 'lb': data.activities[a].lower_bound, 'ub': data.activities[a].upper_bound,'type': data.activities[a].type} for a in durations.keys()})
        # out.append({'durations': durations, 'lbs': {a: data.activities[a].lower_bound for a in durations.keys()}})
    
    eqs = 0
    bests = 0
    for ratio, ratio_opt in zip(ratios, ratios_opt):
        print(f'{ratio:.3f} {ratio_opt:.3f}')
        if ratio == 1.0:
            eqs += 1
        if ratio == ratio_opt:
            bests += 1
    
    print(f'eqs = {eqs}')
    print(f'bests = {bests}')
    print(f'mean ratio: {sum(ratios) / len(ratios)}')
    print(f'mean ratio: {sum(ratios_opt) / len(ratios_opt)}')
    with open('out.pkl', 'wb') as file:
        pickle.dump(out, file)
