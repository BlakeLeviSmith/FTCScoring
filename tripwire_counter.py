"""
TripwireCounter — event-based ball counting via small "tripwire" zones.

Architecture:
  Each alliance has TWO tripwire polygons (drawn by the user):

    1. gate_trip      — placed at the ramp entry (one ball wide).
                        Counts every ball that enters the ramp.
    2. overflow_trip  — placed in the overflow path, past where a
                        classified ball would settle. Counts every ball
                        that rolls all the way through.

  Live scores:
    overflow_total   = overflow_trip.count
    classified_total = gate_trip.count - overflow_trip.count

  Pattern scoring is NOT done live — at end of AUTO/TELEOP the caller
  snapshots the ROI's current ball list and locks it as the pattern.

How counting works (per tripwire — PER-BALL PASS TRACKING):
  We do NOT use boolean "is anything in the zone" — that loses count
  when multiple balls are in the zone at the same time (which happens
  on dense matches).

  Instead, we track each individual ball as it passes through the zone:

    - Each frame, get the set of ball positions inside the tripwire.
    - Try to match each detected ball to an existing "active pass" by
      spatial proximity (Hungarian-greedy nearest match).
    - Matched → update that pass's position + last_seen_frame.
    - Unmatched → it's a NEW pass → count it.
    - Passes that haven't been updated for `persistence_frames` are
      retired (the ball has left the zone).

  This handles 1, 2, 3, or N balls simultaneously in the zone correctly,
  and tolerates short YOLO drops via the persistence window.
"""

import time


class _Tripwire:
    """Single tripwire counter using velocity-projected matching.

    Each frame:
      1. Project every active track forward from its last known position
         using its last known velocity:  pred = last_pos + vel * dt.
      2. Greedily match new detections to nearest projected positions
         within `match_radius_px`. Matched → continuation, no count.
      3. Unmatched new detections start a new track and increment count.
      4. Tracks not seen for `max_misses` frames are retired.

    Velocity is estimated from the last position update (simple finite
    difference) and lightly smoothed. Initial velocity for a brand-new
    track is zero, which means the first match must be within radius
    of the actual last position — fine for the typical "ball enters
    zone, takes 1-2 frames to cross" case.
    """

    def __init__(self, match_radius_px=40, max_misses=3,
                 vel_smooth=0.7):
        self.match_radius_px = match_radius_px
        self.max_misses = max_misses
        self.vel_smooth = vel_smooth  # weight on the new velocity sample
        self.count = 0
        # tid -> {pos:(x,y), vel:(vx,vy), last_seen:int, color}
        self._tracks = {}
        self._next_id = 1
        self._events = []

    def reset(self):
        self.count = 0
        self._tracks = {}
        self._next_id = 1
        self._events = []

    def update(self, balls_in_zone, frame_idx, frame_w_px=1280):
        radius = self.match_radius_px * (frame_w_px / 1280.0)

        # 1. Predict each active track's position this frame.
        predictions = {}
        for tid, t in self._tracks.items():
            dt = frame_idx - t["last_seen"]
            predictions[tid] = (t["pos"][0] + t["vel"][0] * dt,
                                t["pos"][1] + t["vel"][1] * dt)

        # 2. Greedy nearest match: each detection claims at most one track.
        unmatched_dets = list(range(len(balls_in_zone)))
        claimed = set()
        while unmatched_dets:
            best = None  # (dist, det_idx, tid)
            for di in unmatched_dets:
                ball = balls_in_zone[di]
                bx = ball.get("center_x", 0)
                by = ball.get("center_y", 0)
                for tid, (px, py) in predictions.items():
                    if tid in claimed:
                        continue
                    d = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
                    if d > radius:
                        continue
                    if best is None or d < best[0]:
                        best = (d, di, tid)
            if best is None:
                break
            _d, di, tid = best
            ball = balls_in_zone[di]
            new_pos = (ball.get("center_x", 0), ball.get("center_y", 0))
            old = self._tracks[tid]
            pred = predictions[tid]
            dt = max(1, frame_idx - old["last_seen"])
            new_vel = ((new_pos[0] - old["pos"][0]) / dt,
                       (new_pos[1] - old["pos"][1]) / dt)
            # Smooth velocity so a single noisy detection doesn't blow it up.
            a = self.vel_smooth
            self._tracks[tid]["vel"] = (a * new_vel[0] + (1 - a) * old["vel"][0],
                                         a * new_vel[1] + (1 - a) * old["vel"][1])
            self._tracks[tid]["pos"] = new_pos
            self._tracks[tid]["last_seen"] = frame_idx
            self._tracks[tid]["color"] = ball.get("color") or old.get("color")
            claimed.add(tid)
            unmatched_dets.remove(di)
            # Log this match for the debug feed.
            self._log_event({
                "frame": frame_idx,
                "t": time.time(),
                "kind": "matched",
                "tid": tid,
                "color": self._tracks[tid]["color"],
                "pos": [int(new_pos[0]), int(new_pos[1])],
                "pred": [int(pred[0]), int(pred[1])],
                "dist": round(_d, 1),
                "radius": round(radius, 1),
                "count": self.count,
            })

        # 3. Unmatched detections are new balls → new track + count.
        for di in unmatched_dets:
            ball = balls_in_zone[di]
            new_pos = (ball.get("center_x", 0), ball.get("center_y", 0))
            tid = self._next_id
            self._tracks[tid] = {
                "pos": new_pos,
                "vel": (0.0, 0.0),
                "last_seen": frame_idx,
                "color": ball.get("color"),
            }
            self._next_id += 1
            self.count += 1
            # Find the closest active track that was JUST OUT OF reach
            # (would have matched had radius been a bit bigger). This
            # tells you why a count fired when you might not have wanted.
            nearest = None  # (dist, tid)
            for otid, (px, py) in predictions.items():
                if otid in claimed:
                    continue  # already matched to another detection
                d = ((new_pos[0] - px) ** 2 + (new_pos[1] - py) ** 2) ** 0.5
                if nearest is None or d < nearest[0]:
                    nearest = (d, otid)
            self._log_event({
                "frame": frame_idx,
                "t": time.time(),
                "kind": "new",
                "tid": tid,
                "color": ball.get("color"),
                "pos": [int(new_pos[0]), int(new_pos[1])],
                "nearest_dist": round(nearest[0], 1) if nearest else None,
                "nearest_tid": nearest[1] if nearest else None,
                "radius": round(radius, 1),
                "count": self.count,
            })

        # 4. Retire tracks not seen for max_misses frames.
        cutoff = frame_idx - self.max_misses
        retired = [tid for tid, t in self._tracks.items()
                   if t["last_seen"] < cutoff]
        for tid in retired:
            t = self._tracks.pop(tid)
            self._log_event({
                "frame": frame_idx,
                "t": time.time(),
                "kind": "retired",
                "tid": tid,
                "color": t.get("color"),
                "last_pos": [int(t["pos"][0]), int(t["pos"][1])],
                "age": frame_idx - cutoff,
                "count": self.count,
            })

    def _log_event(self, ev):
        ev["seq"] = (self._events[-1]["seq"] + 1) if self._events else 1
        self._events.append(ev)
        if len(self._events) > 300:
            self._events.pop(0)

    def get_active_tracks(self):
        """Snapshot of currently active tracks for the debug panel."""
        return [
            {
                "tid": tid,
                "color": t.get("color"),
                "pos": [int(t["pos"][0]), int(t["pos"][1])],
                "vel": [round(t["vel"][0], 1), round(t["vel"][1], 1)],
                "last_seen": t["last_seen"],
            }
            for tid, t in self._tracks.items()
        ]

    def get_events_since(self, since_seq=0):
        return [e for e in self._events if e.get("seq", 0) > since_seq]


class TripwireCounter:
    """Per-alliance counter holding both tripwires.

    Public API mirrors the old SimpleCountTracker enough that app.py
    swapping is mechanical (set_roi, set_gate_zone-style setters, reset,
    handoff_phase, get_totals, get_exit_events).
    """

    def __init__(self, match_radius_px=40, max_misses=3):
        # match_radius_px: max pixel distance from a track's predicted
        #   position to a new detection for them to be considered the
        #   same ball. Tuned at 1280px frame width; the inner _Tripwire
        #   scales it linearly for the actual frame width.
        # max_misses: how many frames a track can go unseen before it
        #   is retired (bridges short YOLO drops mid-pass).
        self.roi = None
        self.gate_trip_poly = None
        self.overflow_trip_poly = None
        self._frame_idx = 0
        self.match_radius_px = match_radius_px
        self.max_misses = max_misses
        self.gate_trip = _Tripwire(match_radius_px=match_radius_px,
                                    max_misses=max_misses)
        self.overflow_trip = _Tripwire(match_radius_px=match_radius_px,
                                        max_misses=max_misses)
        # Snapshot of cumulative totals at AUTO->TELEOP handoff (display only)
        self._auto_snapshot = None
        # Last list of balls inside the ROI (for end-of-period pattern lock)
        self._last_balls_in_roi = []
        # Rolling history of (frame_idx, balls_in_roi) — used to take a
        # 3-second consensus snapshot for MOTIF pattern at phase end.
        # 60 frames at 15fps ≈ 4s of buffer (a bit of margin).
        from collections import deque
        self._balls_history = deque(maxlen=60)

    def set_match_radius(self, radius_px):
        """Live-update both tripwires' match radius (called from API)."""
        self.match_radius_px = int(radius_px)
        self.gate_trip.match_radius_px = int(radius_px)
        self.overflow_trip.match_radius_px = int(radius_px)

    # ----- config setters (called from app.py at startup + on /api/roi) -----
    def set_roi(self, poly):
        self.roi = poly

    def set_gate_trip(self, poly):
        self.gate_trip_poly = poly

    def set_overflow_trip(self, poly):
        self.overflow_trip_poly = poly

    # Compatibility shims so the old setters from the previous tracker
    # interface don't crash if callers still hit them. They're no-ops.
    def set_gate_zone(self, _): pass
    def set_exit_zone(self, _): pass
    def set_divider(self, _): pass

    # ----- lifecycle -----
    def reset(self):
        self._frame_idx = 0
        self.gate_trip.reset()
        self.overflow_trip.reset()
        self._auto_snapshot = None
        self._last_balls_in_roi = []

    def handoff_phase(self):
        self._auto_snapshot = {
            "gate_count": self.gate_trip.count,
            "overflow_count": self.overflow_trip.count,
        }

    # ----- per-frame update -----
    def update(self, balls, frame_shape):
        """Update both tripwires from a frame's ball list.

        Args:
            balls: list of dicts with center_x, center_y, color (G/P).
                Coordinates in full-frame pixels.
            frame_shape: (h, w, ...) tuple — used to normalize polygons.
        """
        self._frame_idx += 1
        h, w = frame_shape[:2]
        balls = balls or []

        # Filter to balls inside the alliance ROI (sole spatial filter)
        balls_in_roi = [b for b in balls
                        if self._point_in_norm_poly(
                            b.get("center_x", 0), b.get("center_y", 0),
                            self.roi, w, h)]
        self._last_balls_in_roi = balls_in_roi
        # Push NORMALIZED coords into rolling history so the end-of-period
        # consensus sort doesn't have to mix normalized polygons with
        # pixel ball coords (which breaks the projection direction).
        self._balls_history.append((self._frame_idx, [
            {"x_norm": b.get("center_x", 0) / float(w),
             "y_norm": b.get("center_y", 0) / float(h),
             "color":  b.get("color")}
            for b in balls_in_roi
        ]))

        # Gate tripwire — pass the LIST of balls in the zone so each is
        # tracked as a separate pass.
        in_gate = [b for b in balls_in_roi
                   if self._point_in_norm_poly(b.get("center_x", 0),
                                                b.get("center_y", 0),
                                                self.gate_trip_poly, w, h)]
        self.gate_trip.update(in_gate, self._frame_idx, frame_w_px=w)

        # Overflow tripwire
        in_ovr = [b for b in balls_in_roi
                  if self._point_in_norm_poly(b.get("center_x", 0),
                                               b.get("center_y", 0),
                                               self.overflow_trip_poly, w, h)]
        self.overflow_trip.update(in_ovr, self._frame_idx, frame_w_px=w)

        # Return signature kept compatible with old tracker for app.py
        return [], balls_in_roi

    # ----- accessors -----
    def get_totals(self):
        gate = self.gate_trip.count
        ovr = self.overflow_trip.count
        classified = max(0, gate - ovr)
        return {
            "classified": classified,
            "overflow": ovr,
            "occupancy": classified,  # not really live-occupancy any more
            "exited": ovr,            # repurposed: balls that left = overflow passes
            "gate_entries": gate,
        }

    def get_exit_events(self, since_frame=0):
        """Return overflow tripwire 'new' events as 'exit' events for the
        dashboard. Each entry: {frame, t, color}."""
        return [
            {"frame": e["frame"], "t": e["t"], "color": e.get("color")}
            for e in self.overflow_trip._events
            if e.get("kind") == "new" and e["frame"] > since_frame
        ]

    def get_debug_state(self, since_seq=0):
        """Compact snapshot for the dashboard debug panel."""
        return {
            "match_radius_px": self.match_radius_px,
            "frame_idx": self._frame_idx,
            "gate": {
                "count": self.gate_trip.count,
                "active": self.gate_trip.get_active_tracks(),
                "events": self.gate_trip.get_events_since(since_seq.get("gate", 0) if isinstance(since_seq, dict) else 0),
            },
            "overflow": {
                "count": self.overflow_trip.count,
                "active": self.overflow_trip.get_active_tracks(),
                "events": self.overflow_trip.get_events_since(since_seq.get("overflow", 0) if isinstance(since_seq, dict) else 0),
            },
        }

    def get_balls_in_roi(self):
        """Last frame's balls inside the ROI (used at phase end for the
        pattern snapshot)."""
        return list(self._last_balls_in_roi)

    def get_recent_balls_history(self, n_frames=45):
        """Return the most recent n_frames worth of (frame_idx, balls)
        snapshots for the consensus pattern snapshot. Default 45 frames
        ≈ 3s at 15fps."""
        items = list(self._balls_history)
        return items[-n_frames:] if n_frames < len(items) else items

    # No-ops kept for compatibility
    def get_sequence(self): return []
    def get_classified(self): return []
    def get_overflow(self): return []
    def get_occupancy(self): return self.get_totals()["classified"]

    # ----- helpers -----
    @staticmethod
    def _point_in_norm_poly(px, py, poly, w, h):
        """Hit-test a pixel against a normalized polygon.
        Returns False if the polygon is None / too small."""
        if not poly or len(poly) < 3:
            return False
        # Ray-cast point-in-polygon
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
