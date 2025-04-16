import logging
import sys

LOG_FORMAT = "[%(asctime)s][%(levelname)s] [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,  # пишем логи в stderr, чтобы не мешать stdout
)

log = logging.getLogger("app")  # Можно также использовать __name__
