"""BaseLogger class, parent class of other loggers in Cardio."""

import logging
import os
import time
from typing import Any

from tqdm.contrib.logging import logging_redirect_tqdm


class BaseLogger:
    """The logger used within the runner to track metrics."""

    def __init__(
        self,
        cfg: dict | None = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        """Initialise the base logger.

        Base logger that prints to terminal and writes to a file.

        Args:
            cfg (Optional[dict], optional): An dictionary that is
                printed. Defaults to None.
            log_dir (str, optional): The directory to store logs in.
                Defaults to "logs".
            exp_name (str, optional): The name you want to use for the
                individual log files. Defaults to "exp".
            to_file (bool, optional): Whether you want the logs to be
                printed to a file or not. Defaults to True.
        """
        self.file_name = f"{exp_name}_{int(time.time())}"

        if cfg is not None:
            self.terminal(f"Config provided: {cfg}\n")

        if not os.path.exists(os.path.join(log_dir, self.file_name)):
            os.makedirs(os.path.join(log_dir, self.file_name))

        if to_file:
            logging.basicConfig(
                filename=os.path.join(log_dir, self.file_name, "terminal.log"),
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

        self.logger = logging.getLogger()

    def terminal(self, data: Any):
        """Print data to the terminal.

        Method used to send data directly to the terminal, shouldn't be
        used for metrics such as loss, returns etc. but for data like
        configs or messages to the user.

        Args:
            data (Any): Item to be printed.
        """
        with logging_redirect_tqdm():
            self.logger.info(data)

    def log(self, metrics: dict):
        """Send dictionary of metrics to logger.

        Send metrics to the chosen internal logger via a dictionary
        with keys corresponding to the metrics tracked. Used for data
        like loss or evaluation returns.

        Args:
            metrics (dict): Dictionary with metrics to be logged.
        """
        with logging_redirect_tqdm():
            self.logger.info(metrics)
