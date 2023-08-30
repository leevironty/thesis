import matplotlib.pyplot as plt

from argparse import Namespace
import logging
import pathlib
import lightning
from torch_geometric.loader import DataLoader

from thesis.data.wrapper import Data
from thesis import add_log_file_handler
from thesis.evaluation.model import Baseline
from thesis.models.mip.pesp import PESP
from thesis.models.mip.solver import get_solver
from thesis.models.gnn.demo import Demo
from thesis.models.gnn.data import transform
from thesis.evaluation.evaluate import evaluate_weights


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

        obj = evaluate_weights(dataset, weights, solver)
        logger.info(f'(robust eval) Got baseline objective value  {obj}')

    if args.checkpoint:
        logger.info(f'Evaluating the given checkpoint {args.checkpoint}')
        model = Demo.load_from_checkpoint(args.checkpoint)
        transformed = transform(dataset)
        trainer = lightning.Trainer(accelerator='cpu', logger=False)
        out = trainer.predict(model, DataLoader([transformed]))
        assert out is not None

        event_reverse_map = dict(enumerate(dataset.events.keys()))
        weight_dict = {
            (event_reverse_map[u.item()], event_reverse_map[v.item()]): w.item()
            for u, v, w in zip(*transformed['routes'].edge_index, out[0])
        }
        weight_dict = {
            key: max(0, value)
            for key, value in weight_dict.items()
            if key in dataset.activities
        }
        fig, ax = plt.subplots()
        logger.info(f'Len weights: {len(weight_dict)}')
        # ax.hist(weight_dict.values(), bins=50)
        # fig.show()
        # input()
        solver = get_solver(
            log_path='/tmp/thesis_pesp_logs.log',
            threads=args.threads,
            time_limit=120,
        )
        obj = evaluate_weights(data=dataset, weights=weight_dict, solver=solver)
        logger.info(f'Got objective: {obj}')

    if dataset.solution is not None:
        pass
