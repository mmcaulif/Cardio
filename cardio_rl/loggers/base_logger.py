import logging
import os
import time
from typing import Optional

from tqdm.contrib.logging import logging_redirect_tqdm


class BaseLogger:
    def __init__(
        self,
        cfg: Optional[dict] = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        self._exp_name = f"{exp_name}_{int(time.time())}"

        if cfg is not None:
            self.terminal(f"Config provided: {cfg}\n")

        if not os.path.exists(os.path.join(log_dir, self._exp_name)):
            os.makedirs(os.path.join(log_dir, self._exp_name))

        if to_file:
            logging.basicConfig(
                filename=os.path.join(log_dir, self._exp_name, "terminal.log"),
                filemode="w",
                format="%(asctime)s: %(message)s",
                datefmt=" %I:%M:%S %p",
                level=logging.INFO,
            )
        else:
            logging.basicConfig(
                format="%(asctime)s: %(message)s",
                datefmt=" %I:%M:%S %p",
                level=logging.INFO,
            )

        self._logger = logging.getLogger()

    def terminal(self, string):
        with logging_redirect_tqdm():
            self._logger.info(string)

    def log(self, data):
        with logging_redirect_tqdm():
            self._logger.info(data)
