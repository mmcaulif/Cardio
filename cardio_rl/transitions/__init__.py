from .transitions import (BaseTransition,
                          TorchTransition,
                          JaxTransition)

REGISTRY = {}

REGISTRY["numpy"] = BaseTransition
REGISTRY["pytorch"] = TorchTransition
REGISTRY["jax"] = JaxTransition