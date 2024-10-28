import wandb

from cardio_rl.loggers import BaseLogger


class WandbLogger(BaseLogger):
    """Weights and Biases logger that prints to terminal and writes
    to a file and W and B.

    Attributes:
        file_name (str): The name of the file written to. Combines
            the provided exp_name with the current time in seconds.
        _logger (Logger): The python logger used to print to
            terminal.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        """Weights and Biases logger that prints to terminal and writes
        to a file and W and B.

        Args:
            cfg (Optional[dict], optional): An dictionary that is
                printed, in the Wand B logger you must provide a
                config with a project key that corresponds to the
                W and B project you want to log to. Defaults to None.
            log_dir (str, optional): The directory to store logs in.
                Defaults to "logs".
            exp_name (str, optional): The name you want to use for
                the individual log files. Defaults to "exp".
            to_file (bool, optional): Whether you want the logs to
                be printed to a file or not. Defaults to True.

        Raises:
            ValueError: Not providing a cfg dict with a project key.
        """

        super().__init__(cfg, log_dir, exp_name, to_file)

        if (cfg is not None) and ("project" in cfg):
            project_name = cfg["project"]
        else:
            raise ValueError(
                "When using the W and B logger, you must provide a project name via the \
                    cfg argument, e.g. cfg = {'project': <name>} "
            )

        wandb.init(
            project=project_name,
            config=cfg,
        )

    def log(self, data: dict) -> None:
        super().log(data)
        steps = data.pop("Timesteps")
        wandb.log(data, step=steps)
