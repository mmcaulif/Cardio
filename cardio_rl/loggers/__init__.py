"""Default loggers supplied in Cardio."""

# ruff: noqa

from cardio_rl.loggers.base_logger import BaseLogger

try:
    from cardio_rl.loggers.tb_logger import TensorboardLogger
except ImportError:
    pass

try:
    from cardio_rl.loggers.wandb_logger import WandbLogger
except ImportError:
    pass
