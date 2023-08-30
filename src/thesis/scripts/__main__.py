from argparse import ArgumentParser, BooleanOptionalAction
import logging
import os

from thesis.scripts.data_generation import generate_data
from thesis.scripts.evaluation import run_evaluation
from thesis.scripts.train import attach_train_parser
from thesis.scripts.transform_data import attach_run_parser


logger = logging.getLogger('thesis.scripts')


def get_job_sting() -> str:
    job_id = os.environ.get('SLURM_ARRAY_JOB_ID', 'default')
    return f'job_{job_id}'


def get_array_sting() -> str:
    array_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'default')
    return f'array_{array_id}'


def get_threads() -> int:
    return int(os.environ.get('SRUN_CPUS_PER_TASK', 1))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--job-id',
        type=str,
        default=get_job_sting(),
    )
    parser.add_argument(
        '--job-array-id',
        type=str,
        default=get_array_sting(),
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=get_threads(),
    )

    main_subparsers = parser.add_subparsers(required=True, dest='command')

    datagen_parser = main_subparsers.add_parser(
        name='data-gen', help='Generate network - solution pairs.'
    )
    datagen_parser.add_argument('dataset', type=str)
    datagen_parser.add_argument('--count', '-c', type=int)
    datagen_parser.add_argument('--od-share', type=float, default=1.0)
    datagen_parser.add_argument('--out', type=str, default='solutions')
    datagen_parser.add_argument('--seed', type=int, default=123)
    datagen_parser.add_argument(
        '--preprocess', action=BooleanOptionalAction, default=True
    )
    datagen_parser.add_argument('--cycle', default=True, action=BooleanOptionalAction)
    datagen_parser.add_argument('--activity-drop-prob', type=float, default=0.0)
    datagen_parser.add_argument('--time-limit', type=int, default=120)
    datagen_parser.add_argument(
        '--variations', default=True, action=BooleanOptionalAction
    )

    datagen_parser.set_defaults(func=generate_data)

    eval_parser = main_subparsers.add_parser(
        name='evaluate', help='Evaluate produced weights'
    )
    eval_parser.add_argument('--dataset', required=True, type=str)
    eval_parser.add_argument('--baseline', default=True, action=BooleanOptionalAction)
    eval_parser.add_argument('--model', type=str, choices=[])
    eval_parser.add_argument('--out', type=str, default='evaluations')
    eval_parser.add_argument('--time-limit', type=int, default=300)
    eval_parser.add_argument('--checkpoint', type=str)
    eval_parser.set_defaults(func=run_evaluation)

    attach_run_parser(main_subparsers)
    attach_train_parser(main_subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
