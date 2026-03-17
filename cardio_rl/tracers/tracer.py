"""Simple Tracer."""


class Tracer:
    """Base class for tracers that collect transitions."""

    def __init__(self):
        """Initialize the Tracer."""
        self.transition_buffer = []

    @property
    def ready(self):
        """Check if the tracer has collected transitions."""
        return bool(self.transition_buffer)

    def append(self, data: dict):
        """Append a transition to the tracer's buffer."""
        self.transition_buffer.append(data)

    def pop(self):
        """Pop all collected transitions from the tracer's buffer."""
        _transition_buffer = self.transition_buffer.copy()
        self.transition_buffer.clear()
        return _transition_buffer

    def reset(self):
        """Reset the tracer's buffer."""
        self.transition_buffer.clear()
