"""
Temporal smoothing for FTC DECODE scoring.
Provides StableDetector for stabilizing detection output across frames.
"""

from collections import deque
import config


class StableDetector:
    """Temporal smoothing: tracks last N frames for stable detection output."""

    def __init__(self, history_size=None, stability_threshold=None, lock_duration=None):
        hs = history_size or config.STABLE_HISTORY_SIZE
        self.history = deque(maxlen=hs)
        self.stability_threshold = stability_threshold or config.STABLE_THRESHOLD
        self.locked_motif = ""
        self.lock_frames = 0
        self.LOCK_DURATION = lock_duration or config.STABLE_LOCK_DURATION

    def update(self, current_pattern):
        self.history.append(current_pattern)

        if len(self.history) < 3:
            return current_pattern

        pattern_counts = {}
        for p in self.history:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        most_common = max(pattern_counts, key=pattern_counts.get)
        frequency = pattern_counts[most_common] / len(self.history)

        if self.locked_motif and self.lock_frames > 0:
            self.lock_frames -= 1
            if frequency >= 0.8 and most_common != self.locked_motif:
                self.locked_motif = most_common
                self.lock_frames = self.LOCK_DURATION
            return self.locked_motif

        if frequency >= self.stability_threshold:
            self.locked_motif = most_common
            self.lock_frames = self.LOCK_DURATION
            return most_common

        return self.locked_motif if self.locked_motif else most_common

    def reset(self):
        self.history.clear()
        self.locked_motif = ""
        self.lock_frames = 0
