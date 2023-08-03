from thesis.data.wrapper import Data
import pathlib


def test_dataset_format():
    filepath = pathlib.Path(__file__).parent.resolve()
    filepath = filepath.joinpath('sample_datasets/valid')
    Data.from_path(filepath.as_posix())
