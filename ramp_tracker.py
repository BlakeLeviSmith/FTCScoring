"""
RampTracker — entry/exit-aware ball counting for FTC DECODE scoring.

Model
-----
- `gate_zone`  : polygon where a ball ENTERS the RAMP (from the SQUARE side).
                 The first frame a track_id shows up inside this zone, we
                 classify that ball as CLASSIFIED (if ramp has <9) or
                 OVERFLOW (if ramp is already full). Counters are cumulative
                 and never decrement — matches FTC DECODE §10.5.1
                 ("assessment occurs throughout the MATCH").
- `exit_zone`  : polygon where a ball LEAVES the RAMP. When a tracked ball
                 disappears, we check whether its last known position was
                 near the exit zone. If yes → declare EXIT (decrement live
                 occupancy so the next incoming ball can be CLASSIFIED
                 again). If no → assume detector drop, keep occupancy.
- Color-aware grace: purple balls get ~1s before we assume exit; green
                 balls get ~2.5s because the YOLO model loses them more
                 frequently. Configurable in config.py.

Totals
------
`classified_total`, `overflow_total` are cumulative point counters.
`get_occupancy()` returns the LIVE number of classified balls currently
retained on the ramp (used to drive the "ramp full → overflow" latch).
"""

import time
from collections import deque

import cv2
import numpy as np

from config import (
    EXIT_GRACE_SEC_GREEN,
    EXIT_GRACE_SEC_PURPLE,
    EXIT_HARD_TIMEOUT_MULT,
    EXIT_PROXIMITY_MARGIN,
    GATE_COOLDOWN_FRAMES,
    RAMP_MAX_BALLS,
    REBIND_RADIUS,
    REBIND_WINDOW_SEC,
    STABILITY_RADIUS,
    STABILITY_UNLOCK_MOVE,
    STABILITY_WINDOW_SEC,
)


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


class RampTracker:
    """Tracks cumulative classified/overflow counts and live ramp occupancy."""

    def __init__(self):
        self.roi = None
        self.gate_zone = None   # entry zone
        self.exit_zone = None   # exit zone (may be None — falls back to gate)

        # Cumulative counters (points stick even if a ball later leaves).
        self.classified_total = 0
        self.overflow_total = 0
        self.exited_total = 0

        # Live on-ramp state: track_id -> dict(color, last_pos, last_seen, counted_as)
        self._on_ramp = {}

        # IDs we've already counted at the gate, to prevent double-counting
        # a flicker-in/flicker-out on the same track_id.
        self._counted_ids = set()
        self._ids_in_gate = set()

        # Rebind pool: balls whose tracks were lost but might re-acquire
        # under a new ID. Entry: tid -> {color, last_pos, lost_at, counted_as}.
        self._recently_lost = {}

        # Legacy count-delta fallback (when detector provides no track_ids).
        self._cooldown_remaining = 0
        self._prev_gate_count = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_roi(self, roi):
        self.roi = roi

    def set_gate_zone(self, zone):
        self.gate_zone = zone

    def set_exit_zone(self, zone):
        self.exit_zone = zone

    def reset(self):
        self.classified_total = 0
        self.overflow_total = 0
        self.exited_total = 0
        self._on_ramp = {}
        self._counted_ids = set()
        self._ids_in_gate = set()
        self._recently_lost = {}
        self._cooldown_remaining = 0
        self._prev_gate_count = 0

    def handoff_phase(self):
        """Call at phase transitions (e.g. AUTO → TELEOP).

        Balls physically on the ramp carry over to the next phase. We keep
        their live state (so occupancy and pattern scoring stay correct)
        but force them into stable-lock so they're immune to:
          (a) detector drops during the transition pause, and
          (b) being recounted if their track_ids get reassigned (the rebind
              logic catches the new ID and adopts the old entry).

        Cumulative classified/overflow totals are preserved — per the manual,
        artifacts are scored when placed and the points persist.
        """
        now = time.time()
        for entry in self._on_ramp.values():
            entry["stable"] = True
            entry["stable_since"] = entry.get("stable_since", now)
            # Clear motion history so a fresh re-acquisition after the
            # transition doesn't immediately un-lock the ball.
            entry["history"] = deque([(now, entry["last_pos"])], maxlen=64)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, balls, frame_shape):
        if self.roi is None:
            return self.get_sequence(), []

        h, w = frame_shape[:2]
        now = time.time()
        balls = balls or []
        balls_in_roi = [b for b in balls if self._in_region(b, self.roi, w, h)]

        # Prefer ID-based path whenever at least one ball is carrying a track
        # ID. Also run it on empty frames — missing-track evaluation must tick
        # every frame so grace timers expire even during long occlusions.
        any_ids = any(b.get("track_id") is not None for b in balls_in_roi)
        using_ids = any_ids or bool(self._on_ramp) or bool(self._recently_lost)

        if using_ids:
            tracked = [b for b in balls_in_roi if b.get("track_id") is not None]
            self._update_by_id(tracked, now, w, h)
        elif self.gate_zone is not None:
            # Legacy fallback when the detector provides no track_ids at all.
            gate_balls = [b for b in balls_in_roi
                          if self._in_region(b, self.gate_zone, w, h)]
            self._process_gate_by_count(gate_balls)

        return self.get_sequence(), balls_in_roi

    # ------------------------------------------------------------------
    # ID-based entry/exit logic
    # ------------------------------------------------------------------

    def _update_by_id(self, balls_in_roi, now, w, h):
        seen_ids = set()
        rebind_px = REBIND_RADIUS * ((w + h) / 2.0)
        unlock_px = STABILITY_UNLOCK_MOVE * ((w + h) / 2.0)

        for b in balls_in_roi:
            tid = b["track_id"]
            color = b.get("color", "")
            if color not in ("G", "P"):
                continue
            seen_ids.add(tid)
            pos = (b.get("center_x", 0) / w, b.get("center_y", 0) / h)
            pos_px = (pos[0] * w, pos[1] * h)

            if tid in self._on_ramp:
                entry = self._on_ramp[tid]
                # If this ball was stable-locked and suddenly appears far from
                # where it was locked, unlock it — something actually moved.
                if entry.get("stable"):
                    last_px = (entry["last_pos"][0] * w, entry["last_pos"][1] * h)
                    if _dist(pos_px, last_px) > unlock_px:
                        entry["stable"] = False
                        entry["history"] = deque(maxlen=64)
                entry["color"] = color
                entry["last_pos"] = pos
                entry["last_seen"] = now
                entry.setdefault("history", deque(maxlen=64)).append((now, pos))
                self._update_stability(entry, now, w, h)
                continue

            if tid in self._counted_ids:
                # Same track_id we've seen before — not re-countable.
                continue

            # Brand-new track_id. Before treating it as a gate entry, check
            # if it's actually a re-ID of a ball we already know about.
            rebound_from = self._try_rebind(tid, pos, now, color, rebind_px,
                                            w, h)
            if rebound_from is not None:
                continue

            # Position-based dedup: if a known ball (either currently on the
            # ramp and not seen this frame, or in the recently-lost pool) is
            # close by AND that known ball isn't at the gate, assume this
            # new track_id is a re-ID of it and adopt instead of counting.
            # Balls mid-ramp don't silently disappear — if we see a "new"
            # ball there, it's almost always the tracker reassigning IDs.
            # Gate-area positions are excluded because balls genuinely DO
            # appear there as new arrivals.
            adopted = self._try_adopt_by_position(tid, pos, color, rebind_px,
                                                  seen_ids, w, h)
            if adopted:
                continue

            # Genuine new ball. Count any new track_id that appears inside
            # the ROI — we don't require it to be seen crossing the gate.
            # YOLO sometimes gets its first confident detection only after a
            # ball has already settled on the ramp (fast entry between frames,
            # occlusion at the gate, preloaded balls, etc). Requiring a gate
            # sighting meant those balls showed up in the "Balls" count but
            # were never counted as classified/overflow, breaking the
            # invariant `classified + overflow >= live balls`. Occupancy
            # still drives the classified-vs-overflow decision.
            occupancy = self._occupancy()
            if occupancy < RAMP_MAX_BALLS:
                counted_as = "classified"
                self.classified_total += 1
            else:
                counted_as = "overflow"
                self.overflow_total += 1
            self._on_ramp[tid] = {
                "color": color,
                "last_pos": pos,
                "last_seen": now,
                "counted_as": counted_as,
                "stable": False,
                "stable_since": None,
                "history": deque([(now, pos)], maxlen=64),
            }
            self._counted_ids.add(tid)

        # Evaluate missing tracks for exit-vs-drop.
        self._process_missing(seen_ids, now, w, h)
        # Age out the rebind pool.
        self._prune_recently_lost(now)

    def _try_rebind(self, new_tid, pos, now, color, rebind_px, w, h):
        """If `new_tid` is close to a recently-lost ball, adopt that entry.

        Returns the old tid on success, None otherwise. Does NOT increment
        any counters — the ball was already counted when it first arrived.
        """
        best_tid = None
        best_dist = rebind_px + 1
        pos_px = (pos[0] * w, pos[1] * h)
        for old_tid, info in self._recently_lost.items():
            if info["color"] != color:
                continue
            if now - info["lost_at"] > REBIND_WINDOW_SEC:
                continue
            old_px = (info["last_pos"][0] * w, info["last_pos"][1] * h)
            d = _dist(pos_px, old_px)
            if d < best_dist:
                best_dist = d
                best_tid = old_tid
        if best_tid is None:
            return None
        info = self._recently_lost.pop(best_tid)
        self._on_ramp[new_tid] = {
            "color": color,
            "last_pos": pos,
            "last_seen": now,
            "counted_as": info["counted_as"],
            "stable": info.get("stable", False),
            "stable_since": info.get("stable_since"),
            "history": deque([(now, pos)], maxlen=64),
        }
        self._counted_ids.add(new_tid)
        return best_tid

    def _try_adopt_by_position(self, new_tid, pos, color, rebind_px,
                               seen_ids, w, h):
        """If the position is already "owned" by a STABLE-LOCKED known ball
        that's not at the gate, transfer that ball's entry under the new id.

        Only stable-locked balls are adoption anchors. Without this
        constraint, a brand-new overflow ball that rolls onto the ramp can
        silently merge with a temporarily-bumped non-stable neighbor and
        never get counted. Once a ball has been still for STABILITY_WINDOW_SEC
        its position is trusted as "occupied"; any new detection elsewhere
        must be a different physical ball.
        """
        if self._is_in_gate_zone(pos, w, h):
            return False

        # Saturation override: once 9 balls are accounted for, any additional
        # detection must be overflow — we prefer over-counting (ID flicker
        # might cost 1 extra overflow point) over under-counting (missing a
        # real overflow ball worth 1 point anyway). Rebind from _recently_lost
        # below still runs so genuinely flickered-out tracks can be reclaimed.
        if self._occupancy() >= RAMP_MAX_BALLS:
            saturated = True
        else:
            saturated = False

        pos_px = (pos[0] * w, pos[1] * h)
        best_source = None   # ("on_ramp" | "lost", tid)
        best_dist = rebind_px + 1

        if not saturated:
            for tid, entry in self._on_ramp.items():
                if tid == new_tid or tid in seen_ids:
                    continue
                if entry["color"] != color:
                    continue
                if not entry.get("stable"):
                    continue  # only trust locked-in positions as anchors
                last_px = (entry["last_pos"][0] * w, entry["last_pos"][1] * h)
                if self._is_in_gate_zone(entry["last_pos"], w, h):
                    continue  # old position at gate → don't use as anchor
                d = _dist(pos_px, last_px)
                if d < best_dist:
                    best_dist = d
                    best_source = ("on_ramp", tid)

        # Recently-lost adoption is skipped when saturated too — _try_rebind
        # (called before this method) already covered the rebind pool with
        # the same radius, so nothing is actually missed by this branch.
        if not saturated:
            for tid, info in self._recently_lost.items():
                if info["color"] != color:
                    continue
                if not info.get("stable"):
                    continue
                last_px = (info["last_pos"][0] * w, info["last_pos"][1] * h)
                if self._is_in_gate_zone(info["last_pos"], w, h):
                    continue
                d = _dist(pos_px, last_px)
                if d < best_dist:
                    best_dist = d
                    best_source = ("lost", tid)

        if best_source is None:
            return False

        kind, old_tid = best_source
        if kind == "on_ramp":
            old = self._on_ramp.pop(old_tid)
            old["last_pos"] = pos
            old["last_seen"] = time.time()
            self._on_ramp[new_tid] = old
        else:  # "lost"
            info = self._recently_lost.pop(old_tid)
            self._on_ramp[new_tid] = {
                "color": color,
                "last_pos": pos,
                "last_seen": time.time(),
                "counted_as": info["counted_as"],
                "stable": info.get("stable", False),
                "stable_since": info.get("stable_since"),
                "history": deque([(time.time(), pos)], maxlen=64),
            }
        self._counted_ids.add(new_tid)
        return True

    def _is_in_gate_zone(self, pos_norm, w, h):
        if self.gate_zone is None:
            return False
        poly_px = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in self.gate_zone],
            dtype=np.int32,
        )
        px = pos_norm[0] * w
        py = pos_norm[1] * h
        return cv2.pointPolygonTest(poly_px, (float(px), float(py)), False) >= 0

    def _update_stability(self, entry, now, w, h):
        """Mark an entry as stable if its recent history has stayed still."""
        if entry.get("stable"):
            return
        hist = entry.get("history")
        if not hist:
            return
        # Prune history outside the stability window.
        while hist and now - hist[0][0] > STABILITY_WINDOW_SEC:
            hist.popleft()
        if len(hist) < 3:
            return
        if now - hist[0][0] < STABILITY_WINDOW_SEC * 0.9:
            return  # not enough time elapsed yet
        # All positions within STABILITY_RADIUS of the mean?
        radius_px = STABILITY_RADIUS * ((w + h) / 2.0)
        xs = [p[1][0] * w for p in hist]
        ys = [p[1][1] * h for p in hist]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        if all(_dist((x, y), (mx, my)) <= radius_px for x, y in zip(xs, ys)):
            entry["stable"] = True
            entry["stable_since"] = now

    def _prune_recently_lost(self, now):
        expired = [tid for tid, info in self._recently_lost.items()
                   if now - info["lost_at"] > REBIND_WINDOW_SEC]
        for tid in expired:
            self._recently_lost.pop(tid, None)

    def _process_missing(self, seen_ids, now, w, h):
        removed_for_exit = []
        removed_for_rebind = []
        for tid, entry in self._on_ramp.items():
            if tid in seen_ids:
                continue

            # Stable-locked balls are immune from grace-based removal. They
            # only leave when we observe them moving or the match is reset.
            # This stops scores from flickering when robots briefly occlude.
            if entry.get("stable"):
                continue

            grace = (EXIT_GRACE_SEC_GREEN if entry["color"] == "G"
                     else EXIT_GRACE_SEC_PURPLE)
            age = now - entry["last_seen"]
            if age < grace:
                continue

            near_exit = self._near_exit_zone(entry["last_pos"], w, h)
            hard_timeout = age >= grace * EXIT_HARD_TIMEOUT_MULT

            if near_exit or hard_timeout:
                if entry["counted_as"] == "classified":
                    self.exited_total += 1
                removed_for_exit.append(tid)
            else:
                # Not near the exit — probably a detector drop. Stash in the
                # rebind pool so a new track_id near this position adopts it
                # instead of being counted as a fresh gate entry.
                removed_for_rebind.append(tid)

        for tid in removed_for_exit:
            self._on_ramp.pop(tid, None)
        for tid in removed_for_rebind:
            entry = self._on_ramp.pop(tid)
            self._recently_lost[tid] = {
                "color": entry["color"],
                "last_pos": entry["last_pos"],
                "lost_at": now,
                "counted_as": entry["counted_as"],
                "stable": entry.get("stable", False),
                "stable_since": entry.get("stable_since"),
            }

    def _near_exit_zone(self, pos_norm, w, h):
        """True if the ball's last position was inside or within margin of the exit zone.

        Requires an explicitly-set exit_zone. Falling back to the gate zone
        is unsafe: balls sitting in the CLASSIFIED slot closest to the gate
        are always "near the gate", so any occlusion there would be falsely
        declared an exit. Without an exit zone, we treat every missing track
        as a drop (it'll get rebound or age out naturally).
        """
        zone = self.exit_zone
        if zone is None:
            return False
        poly_px = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in zone], dtype=np.int32,
        )
        cx, cy = pos_norm[0] * w, pos_norm[1] * h
        signed_dist = cv2.pointPolygonTest(poly_px, (float(cx), float(cy)), True)
        # signed_dist > 0 inside, 0 on edge, < 0 outside (value = -distance px).
        margin_px = EXIT_PROXIMITY_MARGIN * ((w + h) / 2.0)
        return signed_dist >= -margin_px

    # ------------------------------------------------------------------
    # Legacy fallback (no track_ids)
    # ------------------------------------------------------------------

    def _process_gate_by_count(self, gate_balls):
        current = len(gate_balls)
        if self._cooldown_remaining > 0:
            self._prev_gate_count = current
            self._cooldown_remaining -= 1
            return
        if current > self._prev_gate_count:
            new_count = current - self._prev_gate_count
            gate_balls_sorted = sorted(
                gate_balls, key=lambda b: b.get("confidence", 0), reverse=True,
            )
            new_balls = gate_balls_sorted[self._prev_gate_count:]
            for i in range(new_count):
                if i >= len(new_balls):
                    break
                color = new_balls[i].get("color", "")
                if color not in ("G", "P"):
                    continue
                occ = self._occupancy()
                if occ < RAMP_MAX_BALLS:
                    self.classified_total += 1
                    self._on_ramp[f"legacy_{self.classified_total}"] = {
                        "color": color, "last_pos": (0.5, 0.5),
                        "last_seen": time.time(), "counted_as": "classified",
                    }
                else:
                    self.overflow_total += 1
            self._cooldown_remaining = GATE_COOLDOWN_FRAMES
        self._prev_gate_count = current

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _occupancy(self):
        """Physical occupancy estimate: live on-ramp + presumed-but-lost.

        A ball we just lost detection on hasn't physically left — it's still
        sitting on the ramp. If we don't include `_recently_lost` here,
        occupancy dips during occlusion chaos (overflow balls bumping stable
        balls), the "<9" check lets the NEXT incoming ball through as
        CLASSIFIED, and we end up with 12+ classifieds logged. Counting the
        rebind pool closes that gap — a ball only stops consuming a slot
        once it's either rebound (replaced by a new track_id in the same
        slot) or aged out of the rebind window.
        """
        live = sum(1 for e in self._on_ramp.values()
                   if e["counted_as"] == "classified")
        presumed = sum(1 for info in self._recently_lost.values()
                       if info["counted_as"] == "classified")
        return live + presumed

    def get_occupancy(self):
        return self._occupancy()

    def get_totals(self):
        return {
            "classified": self.classified_total,
            "overflow": self.overflow_total,
            "exited": self.exited_total,
            "occupancy": self._occupancy(),
        }

    def get_sequence(self):
        """Colors of balls currently on the ramp (unsorted).

        Callers sort by spatial position along the ramp axis (in app.py) for
        PATTERN comparison. This is NOT cumulative — it reflects live state.
        """
        return [e["color"] for e in self._on_ramp.values()
                if e["counted_as"] == "classified"]

    def get_classified(self):
        return self.get_sequence()

    def get_overflow(self):
        # Overflow isn't held on the ramp; expose as cumulative count only.
        return []

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
