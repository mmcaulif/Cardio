"""N-step Tracer."""

from collections import deque
from typing import Deque

import numpy as np

from cardio_rl.tracers.tracer import Tracer

# TODO: figure out if this is necessary as a separate module, we
# can achieve something functionally similar by using overlapping
# trajectories and creating some function that squashes the
# trajectories n-step sequences.


class NstepTracer(Tracer):
    """Tracer that collects n-step transitions."""

    def __init__(
        self,
        n_step: int = 1,
    ):
        """Initialize the NstepTracer."""
        self.n_step = n_step
        self.transition_buffer: list = []
        self.nstep_buffer: Deque = deque(maxlen=n_step)

    def append(self, data: dict):
        """Append a transition to the tracer's buffers."""
        self.nstep_buffer.append(data)
        if len(self.nstep_buffer) == self.n_step:
            nstep_transition = {
                "s": self.nstep_buffer[0]["s"],
                "a": self.nstep_buffer[0]["a"],
                "r": np.array([step["r"] for step in self.nstep_buffer]),
                "s_p": self.nstep_buffer[-1]["s_p"],
                "d": self.nstep_buffer[-1]["d"],
            }
            for key, value in self.nstep_buffer[0].items():
                if key not in ["s", "a", "r", "s_p", "d"]:
                    nstep_transition.update({key: value})

            self.transition_buffer.append(nstep_transition)

        if data["d"] and self.n_step > 1:
            self._flush_nstep_buffer()
            self.nstep_buffer.clear()

    def reset(self):
        """Reset the tracer's buffers."""
        self.transition_buffer.clear()
        self.nstep_buffer.clear()

    def _flush_nstep_buffer(self):
        remainder = len(self.nstep_buffer)
        diff = self.n_step - remainder
        if remainder < self.n_step:
            start = 0
        else:
            start = 1

        for i in range(start, remainder):
            temp = list(self.nstep_buffer)[i:]
            pad = [0.0] * (i + diff)  # Ensures reward seq length is fixed to n_steps
            step = {
                "s": temp[0]["s"],
                "a": temp[0]["a"],
                "r": np.array([step["r"] for step in temp] + pad),
                "s_p": temp[-1]["s_p"],
                "d": temp[-1]["d"],
            }

            self.transition_buffer.append(step)
