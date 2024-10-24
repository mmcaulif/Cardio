import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from cardio_rl.loggers import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(
        self,
        cfg: Optional[dict] = None,
        log_dir: str = "logs",
        exp_name: str = "exp",
        to_file: bool = True,
    ) -> None:
        super().__init__(cfg, log_dir, exp_name, to_file)
        tb_log_dir = os.path.join(log_dir, self._exp_name, "tb_logs")
        self.writer = SummaryWriter(tb_log_dir)

    def log(self, data: dict) -> None:
        super().log(data)
        steps = data.pop("Timesteps")
        for key, value in data.items():
            self.writer.add_scalar(key, value, steps)
