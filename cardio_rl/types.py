from typing import TypeAlias

from gymnasium import Env
from gymnasium.experimental.vector import VectorEnv
from numpy.typing import NDArray

Transition: TypeAlias = dict[str, NDArray]
Environment: TypeAlias = Env | VectorEnv
