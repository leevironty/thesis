import matplotlib.pyplot as plt
import torch
import pandas as pd

from argparse import Namespace, BooleanOptionalAction, ArgumentParser
import logging
import pathlib
import lightning
from torch_geometric.loader import DataLoader

import tqdm
from multiprocessing import Pool


from thesis.data.wrapper import Data
from thesis import add_log_file_handler
from thesis.evaluation.model import get_trivial_weights
from thesis.evaluation.evaluate import evaluate_weights
from thesis.models.mip.pesp import PESP
from thesis.models.mip.solver import get_solver
from thesis.models.gnn.hgt import Predictor
from thesis.models.gnn.data import transform


pair = tuple[int, int]

logger = logging.getLogger(__name__)


def attach_eval_parser(main_subparsers):
    eval_parser: ArgumentParser = main_subparsers.add_parser(
        name='evaluate', help='Evaluate produced weights'
    )
    eval_parser.add_argument('--dataset', required=True, type=str)
    eval_parser.add_argument('--share', type=float, default=0.1)
    eval_parser.add_argument('--out', type=str, default='evaluations')
    eval_parser.add_argument('--time-limit', type=int, default=300)
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.set_defaults(func=run_evaluation)


def get_gnn_weights(data: Data, predictor: Predictor) -> dict[pair, float]:
    transformed = transform(data)
    with torch.no_grad():
        out = predictor.predict(transformed)

    weights = {
        (u, v): w.item()
        for (u, v), w in zip(data.activities_constrainable.keys(), out)
    }
    return weights


def get_optimal_weights(data: Data) ->  dict[pair, float]:
    assert data.solution is not None
    weights = data.solution.weights
    weights_with_zeros = {
        key: weights.get(key, 0.0) for key in data.activities_constrainable.keys()
    }
    return weights_with_zeros


def get_evals(data: Data, predictor: Predictor, solver, allow_non_optimal: bool = False, solver_opt = None, solver_gnn = None):
    out = {}
    # get weights
    trivial_weights = get_trivial_weights(data)
    gnn_weights = get_gnn_weights(data, predictor)
    if solver_opt is None:
        solver_opt = solver
    if solver_gnn is None:
        solver_gnn = solver
    out['new'] = evaluate_weights(data, gnn_weights, solver_gnn, allow_non_optimal=allow_non_optimal)
    out['trivial'] = evaluate_weights(data, trivial_weights, solver, allow_non_optimal=allow_non_optimal)
    if data.solution is not None:
        optimal_weights = get_optimal_weights(data)
        max_weight = max(optimal_weights.values())
        # calculate losses
        mse_errors_trivial = [
            (trivial_weights[key] / max_weight - optimal_weights[key] / max_weight) ** 2
            for key in trivial_weights.keys()
        ]
        mse_errors_gnn = [
            (gnn_weights[key] - optimal_weights[key] / max_weight) ** 2
            for key in trivial_weights.keys()
        ]
        mse_loss_trivial = sum(mse_errors_trivial) / len(mse_errors_trivial)
        mse_loss_gnn = sum(mse_errors_gnn) / len(mse_errors_gnn)
        out['optimal'] = evaluate_weights(data, optimal_weights, solver_opt, allow_non_optimal=allow_non_optimal)
        out['optimal_weights'] = optimal_weights
        out['loss_trivial'] = mse_loss_trivial
        out['loss_gnn'] = mse_loss_gnn
        out['gap_trivial'] = out['trivial'] / out['optimal'] - 1
        out['gap_gnn'] = out['new'] / out['optimal'] - 1
    # evaluate weights

    out['weights_trivial'] = trivial_weights
    out['weights_new'] = gnn_weights


    return out



def run_evaluation(args: Namespace):
    output_folder = pathlib.Path(args.out) / args.job_id / args.job_array_id
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / 'logs.log'
    add_log_file_handler(log_path.as_posix())

    logger.info(f'Starting the evaluation process with arguments: {args}')
    logger.info('Reading the evaluation dataset')
    dataset_name = pathlib.Path(args.dataset)
    solver_log_path = output_folder / 'pesp_solver.log'
    solver = get_solver(
        log_path=solver_log_path.as_posix(),
        threads=args.threads,
        time_limit=args.time_limit,
    )
    predictor = Predictor.load_from_checkpoint(args.checkpoint, map_location='cpu')
    if dataset_name.is_dir():
        files = sorted(list(dataset_name.glob('**/solution_*.pkl.gz')))
        index = int(args.share * len(files))
        files = files[-index:]
    else:
        files = [dataset_name]
    results = []
    for i, filename in enumerate(files):
        logger.info(f'Evaluating {i + 1} out of {len(files)}')
        if filename.suffixes == ['.pkl', '.gz']:
            data = Data.from_pickle(filename.as_posix())
        else:
            data = Data.from_path(filename.as_posix())
        try:
            values = get_evals(data, predictor, solver)
            values['filename'] = filename.as_posix()
            results.append(values)
        except RuntimeError:
            logger.warning(f'Skipped file {filename.as_posix()}!')
    df = pd.DataFrame(results)
    numeric_results = df.select_dtypes('number')
    print(f'Means:\n{numeric_results.mean()}')
    print(f'Stds:\n{numeric_results.std()}')
    print(f'Share of zeros:\n{(numeric_results == 0).mean()}')
    print(f'Everything:\n{numeric_results}')
    print(f'Share of new being better:\n{(numeric_results["gap_gnn"] < numeric_results["gap_trivial"]).mean()}')
    print(f'Share of new being at least as good:\n{(numeric_results["gap_gnn"] <= numeric_results["gap_trivial"]).mean()}')
    print(f'Evaluated on {len(results)} / {len(files)} problems')
    df.to_pickle('eval-result.pkl')



    # logger.info('Evaluating baseline solution')
    # trivial_weights = get_trivial_weights(dataset)
    # logger.info('Got weights, running the PESP model')
    # model = PESP(dataset, weights=trivial_weights, solver=solver)
    # model.solve()
    # value = model.objective.value()
    # summary['baseline_objective'] = value
    # logger.info(f'Got baseline objective value {value}')

    # obj = evaluate_weights(dataset, trivial_weights, solver)
    # summary['baseline_robust_objective'] = obj
    # logger.info(f'(robust eval) Got baseline objective value  {obj}')

    # logger.info(f'Evaluating the given checkpoint {args.checkpoint}')
    # predictor.eval()
    # transformed = transform(dataset)
    # trainer = lightning.Trainer(accelerator='cpu', logger=False)
    # out = trainer.predict(model, DataLoader([transformed]))
    # assert out is not None

    # event_reverse_map = dict(enumerate(dataset.events.keys()))
    # weight_dict = {
    #     (event_reverse_map[u.item()], event_reverse_map[v.item()]): w.item()
    #     for u, v, w in zip(*transformed['routes'].edge_index, out[0])
    # }
    # weight_dict = {
    #     key: max(0, value)
    #     for key, value in weight_dict.items()
    #     if key in dataset.activities
    # }
    # fig, ax = plt.subplots()
    # logger.info(f'Len weights: {len(weight_dict)}')
    # # ax.hist(weight_dict.values(), bins=50)
    # # fig.show()
    # # input()
    # solver = get_solver(
    #     log_path='/tmp/thesis_pesp_logs.log',
    #     threads=args.threads,
    #     time_limit=120,
    # )
    # obj = evaluate_weights(data=dataset, weights=weight_dict, solver=solver)
    # summary['gnn_objective'] = obj
    # logger.info(f'Got objective: {obj}')

    # assert dataset.solution is not None

    # optimal_weights = dataset.solution.weights
    # obj = evaluate_weights(data=dataset, weights=optimal_weights, solver=solver)

    # summary['optimal_objective'] = obj

    # logger.info(f'summary:\n{summary}')
