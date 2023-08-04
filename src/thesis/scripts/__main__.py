from argparse import ArgumentParser
import logging

from thesis.scripts.data_generation import generate_data


logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    main_subparsers = parser.add_subparsers(required=True, dest='command')

    datagen_parser = main_subparsers.add_parser(
        name='data-gen',
        help='Generate network - solution pairs.'
    )
    datagen_parser.add_argument('dataset', type=str)
    datagen_parser.add_argument('--count', '-c', type=int)
    datagen_parser.add_argument('--od-share', type=float, default=1.0)
    datagen_parser.add_argument('--out', type=str, default='solutions/test')
    datagen_parser.set_defaults(func=generate_data)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
