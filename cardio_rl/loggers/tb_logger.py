import os

from torch.utils.tensorboard import SummaryWriter

from cardio_rl.loggers import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir="logs", exp_name="exp") -> None:
        super().__init__(log_dir, exp_name)
        tb_log_dir = os.path.join(log_dir, self._exp_name, "tb_logs")
        self.writer = SummaryWriter(tb_log_dir)

    def log(self, data: dict) -> None:
        super().log(data)
        steps = data.pop("Timesteps")
        for key, value in data.items():
            self.writer.add_scalar(key, value, steps)
