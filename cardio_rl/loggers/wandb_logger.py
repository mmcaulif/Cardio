from typing import Optional

import wandb

from cardio_rl.loggers import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project_name: Optional[str] = None,
        cfg: Optional[dict] = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        super().__init__(cfg, log_dir, exp_name, to_file)
        wandb.init(
            project=project_name,
            config=cfg,
        )

    def log(self, data: dict) -> None:
        super().log(data)
        steps = data.pop("Timesteps")
        wandb.log(data, step=steps)
