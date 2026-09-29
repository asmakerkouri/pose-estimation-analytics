"""Exercise repetition counter driven by a joint angle over time (e.g. knee angle for squats)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepCounter:
    down_below: float = 100.0   # angle under which the movement is "down"
    up_above: float = 160.0     # angle above which the movement is "up" again
    smooth: int = 3             # moving-average window against keypoint jitter
    reps: int = 0
    stage: str = "up"
    _buf: list = field(default_factory=list)

    def update(self, angle: float | None) -> int:
        if angle is None:          # keypoint lost for this frame: keep the state
            return self.reps
        self._buf = (self._buf + [angle])[-self.smooth:]
        a = sum(self._buf) / len(self._buf)
        if self.stage == "up" and a < self.down_below:
            self.stage = "down"
        elif self.stage == "down" and a > self.up_above:
            self.stage = "up"
            self.reps += 1
        return self.reps
