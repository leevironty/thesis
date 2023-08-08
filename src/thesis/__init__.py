import logging
import sys


logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(level=logging.DEBUG)
handler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
    fmt='%(levelname)s : %(asctime)s : %(name)s : %(message)s',
)
handler.setFormatter(fmt=formatter)

logger.addHandler(handler)


def add_log_file_handler(path: str):
    file_handler = logging.FileHandler(filename=path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
