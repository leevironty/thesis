from argparse import Namespace, ArgumentParser
import logging
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import time

from torch_geometric.loader import DataLoader
from thesis.models.mip.cycle_basis import TimPassCycle as TimPass
import tqdm
# import torch

from thesis.data.wrapper import Data
from thesis.models.mip.solver import get_solver


logger = logging.getLogger(__name__)

from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    LpInteger,
    LpBinary,
    LpAffineExpression,
    lpSum,
    LpSolver,
    LpStatusOptimal,
)
import random


class MultiSolutionTimPass:
    def __init__(self, data: Data, epsilon: float, solver):
        self.data = data
        self.solver = solver
        self.epsilon = epsilon
        self.original_timpass = TimPass(data, solver)
        self.original_timpass.solve()
        self.rng = random.Random()

    def get_solution(self):
        modified_timpass = self.original_timpass.model.copy()
        old_obj = modified_timpass.objective
        # assert type(old_obj) is LpVariable
        obj_val = old_obj.value()

        modified_timpass.addConstraint(old_obj <= obj_val + self.epsilon)
        modified_timpass.addConstraint(old_obj >= obj_val - self.epsilon)

        # random_preferences = {
        #     act: self.rng.random()
        #     for act in self.data.activities_routable.keys()
        # }
        # random_preferences = {
        #     uv: {
        #         ij: self.rng.random()
        #         for ij in d_.keys()
        #     }
        #     for uv, d_ in self.original_timpass.var_p.items()
        # }
        random_preferences = {
            ij: self.rng.random()
            for ij in self.original_timpass.data.activities_routable.keys()
            # for uv, d_ in self.original_timpass.var_p.items()
        }
        new_obj = lpSum([
            p * random_preferences[ij]
            for uv, d in self.original_timpass.var_p.items()
            for ij, p in d.items()
        ])
        # new_obj = lpSum([
        #     p * self.rng.random()
        #     for d in self.original_timpass.var_p.values()
        #     for p in d.values()
        # ])
        modified_timpass.setObjective(new_obj)
        status = modified_timpass.solve(self.solver)
        if status != LpStatusOptimal:
            raise RuntimeError('Did not find an optimal solution!')
        return random_preferences



def calculate(filename: Path) -> dict:
    data = Data.from_pickle(filename.as_posix())
    solver = get_solver(threads=1, time_limit=20, output=False)
    multisol = MultiSolutionTimPass(data, epsilon=0.0001, solver=solver)
    ws = []
    objectives = []
    for _ in range(20):
        try:
            obj = multisol.get_solution()
        except:
            print('Did not find an optimal solution!')
            continue
        objectives.append(obj.value())
        w = multisol.original_timpass.get_solution().weights
        ws.append(w)
    df = pd.DataFrame(ws).fillna(0)
    expected_max_w = df.max(axis=0).mean()
    total = df.shape[0]
    uniques = df.drop_duplicates().shape[0]
    std = df.T.std(axis=1)
    deviation = std.sum()
    share_changed_weights = (std != 0).mean()
    loss = (df / expected_max_w).var(axis=0).mean()
    return {
        'file': filename.as_posix(),
        'uniques': uniques,
        'total': total,
        'deviation': deviation,
        'share_changed_weights': share_changed_weights,
        'loss': loss,
        'ws': ws,
    }


def attach_ms_checker(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(name='multi-solution', help='Check for multiple solutions')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--share', type=float, required=True)
    parser.set_defaults(func=run_multi_solution_check)


def run_multi_solution_check(args: Namespace):
    logger.info(f'Started batching with args: {args}')

    filenames = sorted(list(Path(args.dataset).glob('**/solution_*.pkl.gz')))
    index = int(len(filenames) * args.share)
    logger.info(f'Will evaluate {index} instances')
    filenames = filenames[:index]

    results = []

    with Pool(processes=args.threads) as pool:
        printouts = 0
        with tqdm.tqdm(total=len(filenames)) as pbar:
            for res in pool.imap_unordered(calculate, filenames):
                results.append(res)
                if printouts <= args.threads:
                    logger.info(f'Done with {res["file"]}')
                    printouts += 1
                pbar.update()
    
    df = pd.DataFrame(results)
    df.to_pickle('multi-solution-evaluation.pkl')
    numeric = df.select_dtypes('number')
    logger.info(f'Got means:\n{numeric.mean()}')
    