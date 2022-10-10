import logging

import tqdm


# https://stackoverflow.com/questions/38543506
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
