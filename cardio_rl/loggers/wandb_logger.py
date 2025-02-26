"""WandbLogger class, inherits from BaseLogger."""

import wandb

from cardio_rl.loggers import BaseLogger


class WandbLogger(BaseLogger):
    """The logger used within the runner to track metrics."""

    def __init__(
        self,
        cfg: dict | None = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        """Initialise the WandbLogger.

        Logger that prints to terminal and writes to a file and Weights
        and Biases.

        Args:
            cfg (Optional[dict], optional): An dictionary that is
                printed, in the Wand B logger you must provide a config
                with a project key that corresponds to the W and B
                project you want to log to Defaults to None.
            log_dir (str, optional): The directory to store logs in.
                Defaults to "logs".
            exp_name (str, optional): The name you want to use for the
                individual log files. Defaults to "exp".
            to_file (bool, optional): Whether you want the logs to be
                printed to a file or not. Defaults to True.

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

    def log(self, metrics: dict) -> None:
        """Send dictionary of metrics to Weights and Biases.

        Send metrics to wandb.log directly via a dictionary with keys
        corresponding to the metrics tracked. Used for data like loss
        or evaluation returns.

        Args:
            metrics (dict): Dictionary with metrics to be logged.
        """
        super().log(metrics)
        steps = metrics.pop("Timesteps")
        wandb.log(metrics, step=steps)
