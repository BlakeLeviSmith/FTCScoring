"""
TripwireCounter — track-ID-based ball counting via small "tripwire" zones.

Architecture (v3 — delegates inter-frame association to ByteTrack):

  Each alliance has TWO tripwire polygons (drawn by the user):
    1. gate_trip      — at the ramp entry (one ball wide).
                        Counts every distinct ball that enters.
    2. overflow_trip  — partway down the overflow path.
                        Counts every distinct ball that rolls through.

  Counting:
    Per frame, ultralytics' tracker (ByteTrack/BoT-SORT — see
    config.YOLO_TRACKER_CONFIG) assigns a persistent track_id to every
    detection. For each detection inside a tripwire polygon, we add its
    track_id to that tripwire's "ever seen" set. The count is just
    len(seen_set).

  No velocity gates, no bounce radii, no Hungarian assignment of our
  own — the tracker handles all of that. The tripwire is just a
  spatial filter that says "this track passed through here."

  Pattern scoring is unchanged — at end of AUTO/TELEOP we sample the
  last 3s of detections inside the ROI and lock the consensus.

Trail visualization:
  We also maintain a per-alliance dict of {track_id: deque[(x, y)]}
  for the last N positions. The detector visualization in app.py uses
  this to draw colored polyline trails on the windowed ROI view —
  long continuous trails = good tracking, multiple short trails for
  what should be one ball = the tracker dropped the id mid-crossing.
"""

import time
from collections import deque

import cv2
import numpy as np


class _Tripwire:
    """Single tripwire counter — counts unique track_ids seen in zone.

    A track_id is added to `seen_ids` the first frame it appears
    inside the polygon. The count is len(seen_ids). Periodically we
    age out track_ids that haven't been seen for memory_frames so a
    later re-issue of the same id (after the original ball is long
    gone) can register as a fresh count.
    """

    def __init__(self, memory_frames=600, **_unused_legacy):
        """Counts unique track_ids appearing in the zone.

        Inter-frame tid continuity (CSRT-loss / re-acquire dedup) is
        handled at the TRACKER level (csrt_tracker.MultiBallTracker)
        via ghost-track resurrection. This counter is intentionally
        dumb: it trusts that the same physical ball keeps the same
        track_id, and just counts distinct ids that crossed the zone.

        `**_unused_legacy` keeps the constructor signature compatible
        with old callers that still pass settled-registry kwargs.
        """
        self.memory_frames = memory_frames
        self.count = 0
        self._seen_ids = {}  # tid -> last frame it appeared in zone
        self._events = []

    def reset(self):
        self.count = 0
        self._seen_ids = {}
        self._events = []

    def update(self, balls_in_zone, frame_idx):
        """Count a track_id the first frame it appears here. Refresh
        last-seen on subsequent appearances so a brief gap inside
        memory_frames doesn't recount."""
        for ball in balls_in_zone:
            tid = ball.get("track_id")
            if tid is None:
                continue
            prev = self._seen_ids.get(tid)
            if prev is None or (frame_idx - prev) > self.memory_frames:
                # New (or memory expired) — count it
                self.count += 1
                self._events.append({
                    "frame": frame_idx,
                    "t": time.time(),
                    "kind": "new",
                    "tid": tid,
                    "color": ball.get("color"),
                    "pos": [int(ball.get("center_x", 0)),
                            int(ball.get("center_y", 0))],
                    "count": self.count,
                })
                if len(self._events) > 200:
                    self._events.pop(0)
            self._seen_ids[tid] = frame_idx

    def get_active_tracks(self, current_frame, recency_frames=15):
        """Track ids seen in this zone within the last N frames."""
        return [tid for tid, last in self._seen_ids.items()
                if (current_frame - last) <= recency_frames]

    def get_events_since(self, since_seq=0):
        # Use frame as proxy for seq (events are append-only)
        return [e for e in self._events
                if e.get("frame", 0) > since_seq]


class TripwireCounter:
    """Per-alliance counter holding both tripwires + trail history."""

    def __init__(self, memory_frames=600, trail_length=30,
                 min_track_age_frames=5,
                 **_unused_legacy):
        self.roi = None
        self.gate_trip_poly = None
        self.overflow_trip_poly = None
        self._frame_idx = 0
        # Skip counting (gate AND overflow) until source_frame reaches
        # this threshold. Used when the early seconds of a clip have an
        # unstable camera, mid-shot pan, etc. — anything that would fire
        # spurious counts before the actual gameplay window. 0 = count
        # from frame 0.
        self.start_frame = 0
        self.memory_frames = memory_frames
        self.trail_length = trail_length
        # Minimum trail length before a track is allowed to count in
        # any tripwire. Lets ByteTrack's tentative→tracked promotion
        # settle so we don't count a track that gets re-issued under a
        # different id 1-2 frames later.
        self.min_track_age_frames = min_track_age_frames
        # First-seen frame per track_id so we can compute age cheaply.
        self._track_first_seen = {}
        self.gate_trip = _Tripwire(memory_frames=memory_frames)
        self.overflow_trip = _Tripwire(memory_frames=memory_frames)
        # Snapshot at AUTO->TELEOP handoff (display only)
        self._auto_snapshot = None
        # Last list of balls inside the ROI (legacy accessor)
        self._last_balls_in_roi = []
        # Rolling history of (frame_idx, balls_in_roi) for end-of-period
        # consensus pattern snapshot. ~60 frames = 2s at 30fps.
        self._balls_history = deque(maxlen=90)
        # Per-track trail history for debug visualization:
        #   {track_id: deque of (center_x, center_y, frame_idx, color)}
        self._trails = {}

    # ----- config setters -----
    def set_roi(self, poly): self.roi = poly
    def set_gate_trip(self, poly): self.gate_trip_poly = poly
    def set_overflow_trip(self, poly): self.overflow_trip_poly = poly

    def set_min_track_age(self, n):
        """Live-update the maturity gate (frames a track must be alive
        before its appearance in a tripwire counts)."""
        self.min_track_age_frames = max(1, int(n))

    # Compatibility shims (old setters from earlier tripwire revisions)
    def set_gate_zone(self, _): pass
    def set_exit_zone(self, _): pass
    def set_divider(self, _): pass
    def set_match_radius(self, _): pass
    def set_initial_gate(self, _): pass
    def set_vel_multiplier(self, _): pass
    def set_bounce_radius(self, _): pass

    # ----- lifecycle -----
    def reset(self):
        self._frame_idx = 0
        self.gate_trip.reset()
        self.overflow_trip.reset()
        self._auto_snapshot = None
        self._last_balls_in_roi = []
        self._balls_history.clear()
        self._trails = {}
        self._track_first_seen = {}

    def handoff_phase(self):
        self._auto_snapshot = {
            "gate_count": self.gate_trip.count,
            "overflow_count": self.overflow_trip.count,
        }

    # ----- per-frame update -----
    def update(self, balls, frame_shape, source_frame=None):
        """source_frame: the SOURCE CLIP frame index. When provided
        (e.g. during replay), events use this instead of the internal
        counter so they align with ground-truth labels recorded with
        _replay_current_frame. When None (live camera), the internal
        counter is used."""
        self._frame_idx += 1
        # Tripwires log events with `source_frame` if available so they
        # match the label-mode coordinate system. Internal counter is
        # the fallback for live camera mode where there's no clip frame.
        log_frame = int(source_frame) if source_frame is not None else self._frame_idx
        h, w = frame_shape[:2]
        balls = balls or []

        # ROI filter
        balls_in_roi = [b for b in balls
                        if self._point_in_norm_poly(
                            b.get("center_x", 0), b.get("center_y", 0),
                            self.roi, w, h)]
        self._last_balls_in_roi = balls_in_roi

        # Push normalized snapshot into rolling history (for end-of-
        # period MOTIF pattern consensus).
        self._balls_history.append((self._frame_idx, [
            {"x_norm": b.get("center_x", 0) / float(w),
             "y_norm": b.get("center_y", 0) / float(h),
             "color":  b.get("color")}
            for b in balls_in_roi
        ]))

        # Update per-track trail history (for debug visualization) AND
        # record first-seen frame per track_id (for maturity check).
        seen_now = set()
        for b in balls_in_roi:
            tid = b.get("track_id")
            if tid is None:
                continue
            seen_now.add(tid)
            if tid not in self._track_first_seen:
                self._track_first_seen[tid] = self._frame_idx
            trail = self._trails.get(tid)
            if trail is None:
                trail = deque(maxlen=self.trail_length)
                self._trails[tid] = trail
            trail.append((float(b.get("center_x", 0)),
                          float(b.get("center_y", 0)),
                          int(self._frame_idx),
                          str(b.get("color") or "?")))
        # Prune trails whose tracks haven't been updated in >2× trail_length
        # frames (long gone — keep memory bounded).
        cutoff = self._frame_idx - self.trail_length * 2
        self._trails = {
            tid: trail for tid, trail in self._trails.items()
            if trail and trail[-1][2] >= cutoff
        }

        # Tripwire counts: filter to in-polygon AND mature track.
        # A track is "mature" once it's been visible in the ROI for at
        # least min_track_age_frames frames. Pre-mature track_ids might
        # still be revised by ByteTrack, so counting them risks
        # double-counting (the same physical ball gets counted under
        # the tentative id, then again under the promoted id).
        min_age = max(1, self.min_track_age_frames)
        def _mature(b):
            tid = b.get("track_id")
            if tid is None:
                return False
            first = self._track_first_seen.get(tid)
            if first is None:
                return False
            return (self._frame_idx - first + 1) >= min_age

        # A ball "transits" a zone if its current position is inside,
        # OR if the segment from its previous detection to its current
        # detection crosses any polygon edge. The segment check catches:
        #   - Fast normal motion that skips over the zone in one frame
        #   - Ghost-resurrected balls that fell THROUGH the zone while
        #     undetected (e.g. ball drops past gate during motion blur,
        #     reappears on the ramp below — its trail's last segment
        #     goes from above-gate to below-gate, crossing the zone)
        def _transits(b, poly):
            cx = b.get("center_x", 0)
            cy = b.get("center_y", 0)
            if self._point_in_norm_poly(cx, cy, poly, w, h):
                return True
            tid = b.get("track_id")
            if tid is None:
                return False
            trail = self._trails.get(tid)
            if not trail or len(trail) < 2:
                return False
            # trail[-1] was just appended above and == current pos;
            # trail[-2] is the prior detected position (possibly many
            # frames ago if CSRT lost & ghost-resurrected the tid).
            px, py, _, _ = trail[-2]
            return self._segment_crosses_norm_poly(
                px, py, cx, cy, poly, w, h)

        # Skip counting entirely while we're before the clip's stable
        # gameplay window (start_frame). Trackers still get refreshed
        # so the visual feed and per-track trails work normally, but
        # no tripwire fires → no inflated counts during camera shifts /
        # pre-match handling.
        in_gate, in_ovr = [], []
        if source_frame is None or source_frame >= self.start_frame:
            in_gate = [b for b in balls_in_roi
                       if _mature(b) and _transits(b, self.gate_trip_poly)]
            in_ovr = [b for b in balls_in_roi
                      if _mature(b) and _transits(b, self.overflow_trip_poly)]
        self.gate_trip.update(in_gate, log_frame)
        self.overflow_trip.update(in_ovr, log_frame)

        return [], balls_in_roi

    # ----- accessors -----
    def get_totals(self):
        gate = self.gate_trip.count
        ovr = self.overflow_trip.count
        classified = max(0, gate - ovr)
        return {
            "classified": classified,
            "overflow": ovr,
            "occupancy": classified,
            "exited": ovr,
            "gate_entries": gate,
        }

    def get_exit_events(self, since_frame=0):
        return [
            {"frame": e["frame"], "t": e["t"], "color": e.get("color")}
            for e in self.overflow_trip._events
            if e.get("frame", 0) > since_frame
        ]

    def get_balls_in_roi(self):
        return list(self._last_balls_in_roi)

    def get_recent_balls_history(self, n_frames=45):
        items = list(self._balls_history)
        return items[-n_frames:] if n_frames < len(items) else items

    def get_trails(self):
        """Return {track_id: list of (x, y, frame, color)} for active
        trails. Used by the visualization layer to draw polylines on
        the windowed ROI view so the user can see where tracks are
        running and where they're being dropped/swapped."""
        return {tid: list(trail) for tid, trail in self._trails.items()}

    # No-ops for backward compat
    def get_sequence(self): return []
    def get_classified(self): return []
    def get_overflow(self): return []
    def get_occupancy(self): return self.get_totals()["classified"]

    def get_debug_state(self, since_seq=0):
        """Compact snapshot for the dashboard debug panel."""
        cur = self._frame_idx
        gate_active = self.gate_trip.get_active_tracks(cur)
        ovr_active = self.overflow_trip.get_active_tracks(cur)
        if isinstance(since_seq, dict):
            sg = int(since_seq.get("gate", 0))
            so = int(since_seq.get("overflow", 0))
        else:
            sg = so = int(since_seq or 0)
        return {
            "frame_idx": cur,
            "tracker": "track_id_based",
            "gate": {
                "count": self.gate_trip.count,
                "active": [{"tid": tid} for tid in gate_active],
                "events": self.gate_trip.get_events_since(sg),
            },
            "overflow": {
                "count": self.overflow_trip.count,
                "active": [{"tid": tid} for tid in ovr_active],
                "events": self.overflow_trip.get_events_since(so),
            },
        }

    # ----- helpers -----
    @staticmethod
    def _point_in_norm_poly(px, py, poly, w, h):
        if not poly or len(poly) < 3:
            return False
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi = poly[i][0] * w
            yi = poly[i][1] * h
            xj = poly[j][0] * w
            yj = poly[j][1] * h
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _segment_crosses_norm_poly(x1, y1, x2, y2, poly, w, h):
        """True if segment (x1,y1)-(x2,y2) intersects any edge of the
        polygon. Used for "did the ball pass THROUGH this zone between
        detections, even if it was never detected INSIDE it" — the
        gate-drop motion-blur case."""
        if not poly or len(poly) < 3:
            return False

        def _ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        def _intersects(ax, ay, bx, by, cx, cy, dx, dy):
            # Standard segment-segment intersection by orientation.
            return (_ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy)
                    and _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy))

        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            xa = poly[i][0] * w
            ya = poly[i][1] * h
            xb = poly[j][0] * w
            yb = poly[j][1] * h
            if _intersects(x1, y1, x2, y2, xa, ya, xb, yb):
                return True
        return False


def stable_color_for_tid(tid):
    """Deterministic, distinct color per track_id. Same id always yields
    the same color so the user can visually trace a track over time."""
    h = (int(tid) * 137) % 180
    hsv = np.uint8([[[h, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
