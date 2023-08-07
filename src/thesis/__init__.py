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

# logging.basicConfig(level=logging.DEBUG)
