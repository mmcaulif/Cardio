

from typing import Any


class Module:
    def step(self, *args: Any, **kwds: Any):
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.step(*args, **kwds)