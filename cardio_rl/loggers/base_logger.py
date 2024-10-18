import logging
import time

from tqdm.contrib.logging import logging_redirect_tqdm


class BaseLogger:
    # TODO: add optional writing to a log file
    def __init__(
        self,
        log_dir="logs",
        exp_name="exp",
    ) -> None:
        self._exp_name = f"{exp_name}_{time.time()}"

        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)

        logging.basicConfig(
            # filename=os.path.join(log_dir, f"{self._exp_name}", "output.log"),
            filemode="w",
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
