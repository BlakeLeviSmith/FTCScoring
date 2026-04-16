"""
SimpleCountTracker — live count + divider-based overflow tracking.

Classified counting:
  Each frame, count colored YOLO detections inside the ROI. The median of
  the last N frames is the live "classified" count. Once that median hits
  CLASSIFIED_MAX (9), snapshot the 9 positions as locked slots (used for
  pattern scoring), and classified stays pinned at 9.

Overflow counting (divider-based):
  A user-drawn 2-point divider splits the ROI into a "classified side"
  (gate side) and an "overflow side". Overflow balls physically roll
  ACROSS the overflow side of the ramp without stopping. For each ball
  detection on the overflow side each frame, we try to match it to an
  active "pass" from the previous frame (same color, nearby position).
  Matched → continue the pass. Unmatched → start a new pass. A pass is
  committed to overflow_total when it hasn't been seen for a short gap.

  The gate centroid tells us which side of the divider line is "overflow"
  (the opposite side from the gate). This requires both a `gate` polygon
  and a `divider` line to be set.

Exits are NOT handled here yet.
No track_ids. No rebind pool.
"""

from collections import deque

import cv2
import numpy as np


class SimpleCountTracker:
    def __init__(self, smoothing_window=15,
                 classified_max=9,
                 pass_match_radius_norm=0.30,
                 pass_gap_frames=8,
                 pass_min_frames=1,
                 overflow_start_fraction=0.33):
        """
        smoothing_window: frames of history for median live count.
        classified_max: 9 per DECODE.
        pass_match_radius_norm: how close (as a fraction of mean(w,h)) a
            new overflow-side detection must be to an active pass for it
            to continue that pass. Generous default — overflow balls move
            fast across the ramp.
        pass_gap_frames: end a pass when it hasn't been seen for this many
            frames. At 15 FPS that's ~0.7s, enough to survive a couple of
            YOLO drops.
        pass_min_frames: a pass must span at least this many total frames
            to be committed as real overflow (kills 1-frame noise).
        overflow_start_fraction: minimum normalized distance along the
            gate→exit axis (0.0 at gate, 1.0 at exit) that an overflow-side
            ball must have reached before it becomes eligible for counting.
            Physically: overflow balls enter at the same height as
            classified balls but don't drop — they keep rolling. We only
            "commit" to calling something overflow once it has passed the
            point where a classified ball would have fallen in. Requires
            both gate_zone and exit_zone to be set.
        """
        self.roi = None
        self.gate_zone = None
        self.exit_zone = None
        self.divider = None

        self.smoothing_window = smoothing_window
        self.classified_max = classified_max
        self.pass_match_radius_norm = pass_match_radius_norm
        self.pass_gap_frames = pass_gap_frames
        self.pass_min_frames = pass_min_frames
        self.overflow_start_fraction = overflow_start_fraction

        # Per-frame state
        self._count_hist = deque(maxlen=smoothing_window)
        self._last_sequence = []
        self._last_balls_in_roi = []
        self._frame_idx = 0

        # Classified slot lock for pattern scoring.
        self._classified_slots = []   # [(x_norm, y_norm), ...]
        self._slots_locked = False

        # Overflow passes currently in flight.
        # id -> {color, last_pos_px, last_frame, first_frame}
        self._active_passes = {}
        self._next_pass_id = 1

        # Recently-committed passes. A short window after a pass closes,
        # any new detection within match_radius of where it last was gets
        # treated as the same ball (prevents end-of-ramp double-counts
        # from a trailing straggler detection after a brief YOLO drop).
        # entry: {color, last_pos_px, committed_frame}
        self._recently_committed = []
        # Same unit as pass_gap_frames; kept separate so users can widen
        # the lockout without widening the in-flight tolerance.
        self.post_commit_lockout_frames = 20

        # Cumulative overflow commits.
        self.overflow_total = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_roi(self, roi):
        self.roi = roi

    def set_gate_zone(self, zone):
        self.gate_zone = zone

    def set_exit_zone(self, zone):
        self.exit_zone = zone

    def set_divider(self, line):
        """Two-point line separating classified from overflow zones along
        the ramp. Stored but not yet used for scoring — scoring behavior
        against the divider will be wired in after you mark it and
        confirm how you want it interpreted.
        """
        self.divider = line if (line and len(line) == 2) else None

    def reset(self):
        self._count_hist.clear()
        self._last_sequence = []
        self._last_balls_in_roi = []
        self._frame_idx = 0
        self._classified_slots = []
        self._slots_locked = False
        self._active_passes = {}
        self._next_pass_id = 1
        self._recently_committed = []
        self.overflow_total = 0

    def handoff_phase(self):
        """AUTO → TELEOP handoff. Keep slots locked; nothing to reset."""
        pass

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, balls, frame_shape):
        if self.roi is None:
            return [], []

        self._frame_idx += 1
        h, w = frame_shape[:2]
        balls_in_roi = [b for b in (balls or [])
                        if self._in_region(b, self.roi, w, h)]
        colored = [b for b in balls_in_roi
                   if b.get("color") in ("G", "P")]

        self._count_hist.append(len(colored))
        self._last_sequence = [b["color"] for b in colored]
        self._last_balls_in_roi = balls_in_roi

        if not self._slots_locked:
            self._maybe_lock_slots(colored, w, h)
        self._update_overflow_passes(colored, w, h)

        return list(self._last_sequence), balls_in_roi

    # ------------------------------------------------------------------
    # Slot lock (for pattern scoring)
    # ------------------------------------------------------------------

    def _maybe_lock_slots(self, colored, w, h):
        """Lock the 9 classified slots as soon as we've stably seen 9 balls."""
        if self._smoothed_count() < self.classified_max:
            return
        if len(colored) < self.classified_max:
            return
        chosen = sorted(colored, key=lambda b: b.get("center_x", 0))[:self.classified_max]
        self._classified_slots = [
            (b["center_x"] / w, b["center_y"] / h) for b in chosen
        ]
        self._slots_locked = True

    # ------------------------------------------------------------------
    # Overflow passes (divider-based)
    # ------------------------------------------------------------------

    def _update_overflow_passes(self, colored, w, h):
        """Track balls rolling through the overflow side of the divider.

        Only needs `divider` (a 2-point line). The classified side is
        defined by GRAVITY: whichever side of the divider is lower in the
        image (larger y). This handles slanted dividers naturally — a
        tilted line still correctly separates "uphill" from "downhill"
        regardless of its angle, and it doesn't depend on where the Gate
        polygon was drawn.
        """
        if self.divider is None:
            return

        classified_sign = self._divider_sign_of_ground(w, h)
        if classified_sign is None:
            return  # divider too short / degenerate

        match_radius = self.pass_match_radius_norm * (w + h) / 2.0
        axis = self._gate_to_exit_axis(w, h)  # None if no exit_zone set

        # Filter to balls that are eligible for overflow counting
        # (overflow side of divider + past the start-fraction threshold).
        eligible = []
        for b in colored:
            px = b.get("center_x", 0)
            py = b.get("center_y", 0)
            sign = self._divider_sign(px, py, w, h)
            if sign == 0 or sign == classified_sign:
                continue
            if axis is not None:
                t = self._project_axis_fraction(px, py, axis)
                if t < self.overflow_start_fraction:
                    continue
            eligible.append((b, (px, py)))

        # One-to-one assignment: each pass can be claimed by at most one
        # detection per frame. Without this, two distinct overflow balls
        # at similar positions both update the same pass and the second
        # is silently lost. We greedily pick the (detection, pass) pair
        # with the smallest distance, claim them, and repeat.
        claimed_passes = set()
        unmatched = list(range(len(eligible)))
        while unmatched:
            best = None  # (dist, det_idx, pass_id)
            for di in unmatched:
                ball, pos = eligible[di]
                color = ball.get("color", "")
                for pid, info in self._active_passes.items():
                    if pid in claimed_passes:
                        continue
                    if info["color"] != color:
                        continue
                    d = _dist(pos, info["last_pos_px"])
                    if d > match_radius:
                        continue
                    if best is None or d < best[0]:
                        best = (d, di, pid)
            if best is None:
                break
            _d, di, pid = best
            ball, pos = eligible[di]
            info = self._active_passes[pid]
            info["last_pos_px"] = pos
            info["last_frame"] = self._frame_idx
            claimed_passes.add(pid)
            unmatched.remove(di)

        # Whatever's left starts a new pass (or hits the post-commit lockout).
        for di in unmatched:
            ball, pos = eligible[di]
            self._start_new_pass_or_suppress(ball, pos, match_radius)

        self._commit_expired_passes()

    def _start_new_pass_or_suppress(self, ball, pos_px, match_radius):
        """Open a new pass at (ball, pos_px), unless this position is within
        the post-commit lockout zone of a recently-committed pass."""
        color = ball.get("color", "")
        for entry in self._recently_committed:
            if entry["color"] != color:
                continue
            age = self._frame_idx - entry["committed_frame"]
            if age > self.post_commit_lockout_frames:
                continue
            if _dist(pos_px, entry["last_pos_px"]) <= match_radius:
                entry["last_pos_px"] = pos_px
                entry["committed_frame"] = self._frame_idx
                return  # suppressed
        self._active_passes[self._next_pass_id] = {
            "color": color,
            "last_pos_px": pos_px,
            "first_frame": self._frame_idx,
            "last_frame": self._frame_idx,
        }
        self._next_pass_id += 1

    def _commit_expired_passes(self):
        expired = [pid for pid, info in self._active_passes.items()
                   if (self._frame_idx - info["last_frame"]) > self.pass_gap_frames]
        for pid in expired:
            info = self._active_passes.pop(pid)
            span = info["last_frame"] - info["first_frame"] + 1
            if span >= self.pass_min_frames:
                self.overflow_total += 1
                self._recently_committed.append({
                    "color": info["color"],
                    "last_pos_px": info["last_pos_px"],
                    "committed_frame": self._frame_idx,
                })
        # Prune the recently-committed list to its lockout window.
        cutoff = self._frame_idx - self.post_commit_lockout_frames
        self._recently_committed = [
            e for e in self._recently_committed
            if e["committed_frame"] >= cutoff
        ]

    # Sign of (px, py) relative to the divider line using the 2D cross
    # product of (p2 - p1) × (point - p1). Returns +1, 0, or -1.
    def _divider_sign(self, px, py, w, h):
        (x1n, y1n), (x2n, y2n) = self.divider
        x1, y1 = x1n * w, y1n * h
        x2, y2 = x2n * w, y2n * h
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def _divider_sign_of_gate(self, w, h):
        """Retained for reference; no longer used by the overflow path."""
        if not self.gate_zone or len(self.gate_zone) < 3:
            return None
        xs = [p[0] for p in self.gate_zone]
        ys = [p[1] for p in self.gate_zone]
        gx = sum(xs) / len(xs) * w
        gy = sum(ys) / len(ys) * h
        sign = self._divider_sign(gx, gy, w, h)
        return sign if sign != 0 else None

    def _divider_sign_of_ground(self, w, h):
        """Sign of a point "below" the divider midpoint in image coords.

        Image y grows downward, so a point (midx, midy + large_offset) is
        closer to the ground. Its sign wrt the divider line identifies the
        classified side (the side where balls rest because gravity pulls
        them there). This works for any divider angle — a slanted line
        still has a well-defined "below" side, because we pick a probe
        point far enough away that floating-point wiggle is irrelevant.
        """
        if not self.divider or len(self.divider) != 2:
            return None
        (x1n, y1n), (x2n, y2n) = self.divider
        x1, y1 = x1n * w, y1n * h
        x2, y2 = x2n * w, y2n * h
        # Reject near-horizontal zero-length lines.
        if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
            return None
        midx = (x1 + x2) / 2.0
        midy = (y1 + y2) / 2.0
        # Probe point 10× the frame height below the midpoint — unambiguously
        # on the ground side.
        probe_x = midx
        probe_y = midy + h * 10.0
        sign = self._divider_sign(probe_x, probe_y, w, h)
        return sign if sign != 0 else None

    def _centroid_px(self, poly, w, h):
        if not poly or len(poly) < 3:
            return None
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs) / len(xs) * w, sum(ys) / len(ys) * h)

    def _gate_to_exit_axis(self, w, h):
        """Return (gate_pt_px, axis_vec_px, length_sq_px) or None.

        The axis runs from gate centroid to exit centroid. Returns None
        unless both polygons are set and non-degenerate.
        """
        gate_c = self._centroid_px(self.gate_zone, w, h)
        exit_c = self._centroid_px(self.exit_zone, w, h)
        if gate_c is None or exit_c is None:
            return None
        vx = exit_c[0] - gate_c[0]
        vy = exit_c[1] - gate_c[1]
        length_sq = vx * vx + vy * vy
        if length_sq < 1.0:
            return None
        return (gate_c, (vx, vy), length_sq)

    def _project_axis_fraction(self, px, py, axis):
        """Project (px, py) onto the gate→exit axis, return t in [~0, ~1]."""
        (gx, gy), (vx, vy), length_sq = axis
        dx = px - gx
        dy = py - gy
        return (dx * vx + dy * vy) / length_sq

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _smoothed_count(self):
        if not self._count_hist:
            return 0
        s = sorted(self._count_hist)
        return int(s[len(s) // 2])

    def get_totals(self):
        classified = (len(self._classified_slots)
                      if self._slots_locked
                      else self._smoothed_count())
        # Active (not-yet-committed) passes count toward the live display
        # so overflow ticks up while a ball is still crossing the zone.
        # They're only added to `overflow_total` once they expire.
        active = sum(
            1 for info in self._active_passes.values()
            if (self._frame_idx - info["first_frame"] + 1) >= self.pass_min_frames
        )
        return {
            "classified": classified,
            "overflow": self.overflow_total + active,
            "exited": 0,
            "occupancy": classified,
        }

    def get_sequence(self):
        return list(self._last_sequence)

    def get_classified(self):
        return self.get_sequence()

    def get_overflow(self):
        return []

    def get_occupancy(self):
        return self.get_totals()["classified"]

    # ------------------------------------------------------------------
    # Region helper
    # ------------------------------------------------------------------

    @staticmethod
    def _in_region(ball, region, frame_w, frame_h):
        cx = ball.get("center_x", 0)
        cy = ball.get("center_y", 0)
        if region and isinstance(region[0], (list, tuple)):
            poly_px = np.array(
                [[int(p[0] * frame_w), int(p[1] * frame_h)] for p in region],
                dtype=np.int32,
            )
            return cv2.pointPolygonTest(poly_px, (float(cx), float(cy)), False) >= 0
        x1 = region[0] * frame_w
        y1 = region[1] * frame_h
        x2 = region[2] * frame_w
        y2 = region[3] * frame_h
        return x1 <= cx <= x2 and y1 <= cy <= y2


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5
