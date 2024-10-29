import os

from torch.utils.tensorboard import SummaryWriter

from cardio_rl.loggers import BaseLogger


class TensorboardLogger(BaseLogger):
    """Tensorboard logger that prints to terminal and writes to a file and
    tensorboard.

    Attributes:
        file_name (str): The name of the file written to. Combines
            the provided exp_name with the current time in seconds.
        _logger (Logger): The python logger used to print to
            terminal.
        writer (Summarywriter): The tensorboard Summarywriter used
            to log metrics.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        """Tensorboard logger that prints to terminal and writes to a file and
        tensorboard.

        Args:
            cfg (Optional[dict], optional): An dictionary that is
                printed. Defaults to None.
            log_dir (str, optional): The directory to store logs in.
                Defaults to "logs".
            exp_name (str, optional): The name you want to use for
                the individual log files. Defaults to "exp".
            to_file (bool, optional): Whether you want the logs to
                be printed to a file or not. Defaults to True.
        """
        super().__init__(cfg, log_dir, exp_name, to_file)
        tb_log_dir = os.path.join(log_dir, self.file_name, "tb_logs")
        self.writer = SummaryWriter(tb_log_dir)

    def log(self, metrics: dict) -> None:
        """Send dictionary of metrics to Tensorboard based on the keys and
        values.

        Args:
            metrics (dict): Dictionary with metrics to be logged.
        """
        super().log(metrics)
        steps = metrics.pop("Timesteps")
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, steps)
