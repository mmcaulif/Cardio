from typing import Optional

import wandb

from cardio_rl.loggers import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self, project_name, log_dir="logs", exp_name="exp", cfg: Optional[dict] = None
    ) -> None:
        super().__init__(log_dir, exp_name)
        wandb.init(
            project=project_name,
            config=cfg,
        )

    def log(self, data: dict) -> None:
        super().log(data)
        steps = data.pop("Timesteps")
        wandb.log(data, step=steps)
