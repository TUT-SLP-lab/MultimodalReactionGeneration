"""Generate logger"""

from logging import getLogger, StreamHandler, FileHandler, Formatter, Logger, WARN
from datetime import datetime


class DummyLogger:
    def info(self, *_a, **_k) -> None:
        return


def set_logger(name, rootname="log/main.log", use_handler=False) -> Logger:
    """generate logger"""
    dt_now = datetime.now()
    dtime = dt_now.strftime("%Y%m%d_%H%M%S")
    fname = rootname + "." + dtime
    logger = getLogger(name)

    if use_handler:
        handler1 = StreamHandler()
        handler1.setFormatter(
            Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        handler1.setLevel(WARN)
        logger.addHandler(handler1)

        handler2 = FileHandler(filename=fname)
        handler2.setFormatter(
            Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        handler2.setLevel(WARN)  # handler2 more than Level.WARN
        logger.addHandler(handler2)

    return logger
