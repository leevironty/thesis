


from argparse import Namespace, ArgumentParser
import pandas as pd
import numpy as np
import random

from thesis.models.gnn.hgt import Predictor
from thesis.data.wrapper import Data
from thesis.models.gnn.data import transform
from thesis.models.mip.solver import get_solver
from thesis.scripts.evaluation import get_evals
import pickle



def attach_big_eval(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(name='big-eval', help='Evaluate large problem instances')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--time-limit-gnn', type=int, required=True)
    parser.add_argument('--rel-gap-gnn', type=float, required=True)
    parser.add_argument('--time-limit-trivial', type=int, required=True)
    parser.add_argument('--rel-gap-trivial', type=float, required=True)
    parser.set_defaults(func=run_big_eval)


def run_big_eval(args: Namespace):
    gnn = Predictor.load_from_checkpoint(args.checkpoint, map_location='cpu')
    data = Data.from_path(args.dataset_path)
    rng = random.Random(1)

    prefs = {
        act: rng.random()
        for act in data.activities_routable.keys()
    }
    data.preferences = prefs

    solver_trivial = get_solver(threads=args.threads, time_limit=args.time_limit_trivial, output=True, rel_gap=args.rel_gap_trivial)
    solver_gnn = get_solver(threads=args.threads, time_limit=args.time_limit_gnn, output=True, rel_gap=args.rel_gap_gnn)
    eval_out = get_evals(data, gnn, solver=solver_trivial, solver_gnn=solver_gnn, allow_non_optimal=True)
    with open('big-eval-out.pkl', 'wb') as file:
        pickle.dump(eval_out, file)
