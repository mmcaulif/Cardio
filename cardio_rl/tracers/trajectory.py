"""Trajectory Tracer."""

from collections import deque
from typing import Deque

from cardio_rl.tracers.tracer import Tracer
from cardio_rl.tree import stack


class TrajectoryTracer(Tracer):
    """Tracer that collects trajectories of fixed length."""

    def __init__(self, seq_len: int = 1, period: int = 1, overlapping: bool = True):
        """Initialize the TrajectoryTracer."""
        self.seq_len = seq_len
        self.period = period
        self.overlapping = overlapping
        self.transition_buffer: list = []
        self.trajectory_buffer: Deque = deque(maxlen=seq_len)
        self.i = 0

    def append(self, data: dict):
        """Append a transition to the tracer's buffers."""
        self.trajectory_buffer.append(data)
        if len(self.trajectory_buffer) == self.seq_len:
            stacked_trajectory = stack(list(self.trajectory_buffer))
            if self.i % self.period == 0:
                self.transition_buffer.append(stacked_trajectory)
            self.i += 1

        if not self.overlapping and data["d"]:
            self.trajectory_buffer.clear()

    def reset(self):
        """Reset the tracer's buffers."""
        self.transition_buffer.clear()
        self.trajectory_buffer.clear()
