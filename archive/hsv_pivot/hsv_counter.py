"""
HSV blob counter — pivot from YOLO/CSRT tracking to direct HSV blob detection.

Tracks TWO colors (purple + green) over ONE counting line. Calibration:
the user draws a circle on the ball; the circle defines both the HSV
sample area (mean of all pixels inside) and the per-ball pixel size
(π · r²). The first calibration also sets the shared ball_area_px;
subsequent calibrations just refresh that color's HSV.

Per-frame counting per color:
  1. HSV mask (bounded by the calibrated tolerance).
  2. Connected components → list of blobs (centroid + area).
  3. Match each blob to the closest previous-frame blob of that color.
  4. If the segment between centroids crosses the user's line,
     count += round(blob_area / ball_area_px).
"""

import json
import os
import threading
import time

import cv2
import numpy as np


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hsv_config.json")

COLORS = ("purple", "green")
COLOR_BGR = {                        # for annotation
    "purple": (255, 0, 200),
    "green":  (0, 255, 100),
}

# Two counting lines: "classified" at the gate (entry), "overflow"
# further down the ramp. A crossing on classified counts the ball as
# entered; a crossing on overflow counts it as overflowed. Score is
# computed elsewhere as classified - overflow.
LINES = ("classified", "overflow")
LINE_BGR = {
    "classified": (0, 200, 255),   # orange
    "overflow":   (255, 200, 60),  # cyan
}


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    def ccw(px, py, qx, qy, rx, ry):
        return (ry - py) * (qx - px) > (qy - py) * (rx - px)
    return (ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy)
            and ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy))


class HsvBlobCounter:
    """Two-color, single-line counter."""

    def __init__(self, hsv_tol=(15, 70, 70), match_radius_px=250,
                 mask_close_radius=4,
                 aspect_min=0.5, aspect_max=2.0,
                 cooldown_frames=5, cooldown_radius_mult=1.0):
        self._lock = threading.Lock()
        # Per-color HSV center (h, s, v) — MEDIAN of pixels inside the
        # user's calibration circle. Median (not mean) so the rod pixels
        # cutting across the ball don't drag the value toward gray.
        self.hsv = {c: None for c in COLORS}
        self.hsv_tol = hsv_tol
        # Morphological closing kernel radius — bridges gaps in the
        # mask. The rod running through balls creates a thin un-matched
        # stripe; closing with a radius >= half the rod width re-merges
        # the two halves of each ball into one connected component.
        # Default 4 → closes gaps up to ~9 px wide.
        self.mask_close_radius = int(mask_close_radius)
        # Aspect-ratio filter: reject blobs whose width/height is outside
        # [min, max]. Catches motion-blur trails (very tall + thin or
        # very wide + short) that would otherwise be counted by area.
        # 0.5 .. 2.0 = roughly square to "twice as tall as wide".
        self.aspect_min = float(aspect_min)
        self.aspect_max = float(aspect_max)
        # Spatial+temporal cooldown after a crossing fires. For
        # cooldown_frames frames, any new crossing whose centroid is
        # within (ball_diameter * cooldown_radius_mult) px of a
        # recently-counted crossing is suppressed. Stops the same
        # physical ball's motion trail / shadow / reflection from
        # registering as multiple separate crossings on consecutive
        # frames.
        self.cooldown_frames = int(cooldown_frames)
        self.cooldown_radius_mult = float(cooldown_radius_mult)
        # Per-line list of recently-fired crossings: [(cx, cy, frame_idx)]
        self._recent_crossings = {n: [] for n in LINES}
        # Shared ball pixel area, set by FIRST calibration.
        self.ball_area_px = None
        # Two counting lines in source-image coords.
        self.lines = {n: None for n in LINES}
        # Auto counts: per-line, per-color.
        self.count = {n: {c: 0 for c in COLORS} for n in LINES}
        # MANUAL ground-truth counts. User clicks +1 every time they see
        # a ball cross. Persisted to hsv_config.json so the slow human
        # scoring pass only has to happen once per clip.
        self.manual_count = {n: {c: 0 for c in COLORS} for n in LINES}
        # Per-click event log: each manual click records frame index +
        # wall-clock timestamp + line/color. Enables timeline-based
        # comparison ("at frame 1234 user saw a ball cross; did auto
        # also fire a crossing within ±15 frames?") instead of just
        # total-count comparison.
        self.manual_events = []
        self._manual_seq = 0
        # Path of the video the manual count was collected against
        # (loaded from config). Used to warn the user if they switch clips.
        self.manual_source_video = None
        # Previous-frame blob positions per color (shared across lines).
        self._prev_blobs = {c: [] for c in COLORS}
        self.match_radius_px = match_radius_px
        # Last debug snapshot
        self.last_blob_count = {c: 0 for c in COLORS}
        # Ring buffer of recent crossings — what the dashboard needs to
        # answer "why did the count jump by 4 when only 2 balls passed?".
        self.crossings = []
        self._cross_seq = 0
        # "normal" = original + mask tint; "mask" = black bg, only matched
        # pixels visible. Toggleable so user can see exactly what HSV is
        # picking up.
        self.view_mode = "normal"
        # Frames in which blobs that just crossed get highlighted yellow.
        self._highlight_frames = 6
        self._frame_idx = 0
        self._load()

    def set_tolerance(self, dh, ds, dv):
        with self._lock:
            self.hsv_tol = (int(dh), int(ds), int(dv))

    def set_view_mode(self, mode):
        if mode in ("normal", "mask"):
            self.view_mode = mode

    def set_close_radius(self, r):
        with self._lock:
            self.mask_close_radius = max(0, int(r))

    def set_aspect_filter(self, lo, hi):
        with self._lock:
            self.aspect_min = max(0.0, float(lo))
            self.aspect_max = max(self.aspect_min + 0.01, float(hi))

    def set_cooldown(self, frames, radius_mult):
        with self._lock:
            self.cooldown_frames = max(0, int(frames))
            self.cooldown_radius_mult = max(0.0, float(radius_mult))

    # ---------------- persistence ----------------
    def _load(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH) as f:
                d = json.load(f)
            for c in COLORS:
                v = (d.get("hsv") or {}).get(c)
                if v:
                    self.hsv[c] = tuple(v)
            self.ball_area_px = d.get("ball_area_px")
            lines = d.get("lines") or {}
            for n in LINES:
                ln = lines.get(n)
                if ln:
                    self.lines[n] = (tuple(ln[0]), tuple(ln[1]))
            # Restore previously-recorded ground-truth so live scoring
            # only has to happen once per clip.
            mc = d.get("manual_count") or {}
            for n in LINES:
                for c in COLORS:
                    self.manual_count[n][c] = int((mc.get(n) or {}).get(c, 0))
            self.manual_source_video = d.get("manual_source_video")
            self.manual_events = list(d.get("manual_events") or [])
            self._manual_seq = max((e.get("seq", 0)
                                    for e in self.manual_events),
                                   default=0)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def _save(self):
        d = {
            "hsv": {c: list(self.hsv[c]) if self.hsv[c] else None
                    for c in COLORS},
            "ball_area_px": self.ball_area_px,
            "lines": {n: [list(self.lines[n][0]), list(self.lines[n][1])]
                          if self.lines[n] else None
                      for n in LINES},
            "manual_count": {n: dict(self.manual_count[n]) for n in LINES},
            "manual_events": list(getattr(self, "manual_events", [])),
            "manual_source_video": getattr(self, "manual_source_video", None),
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(d, f, indent=2)

    # ---------------- public API ----------------
    @property
    def calibrated_any(self):
        return any(self.hsv[c] is not None for c in COLORS)

    @property
    def calibrated_all(self):
        return (self.ball_area_px is not None
                and all(self.hsv[c] is not None for c in COLORS))

    def set_line(self, which, p1, p2):
        if which not in LINES:
            return False
        with self._lock:
            self.lines[which] = (tuple(p1), tuple(p2))
            for c in COLORS:
                self._prev_blobs[c] = []
            self._save()
        return True

    def increment_manual(self, which, color, delta, frame_idx=None):
        """Record a +1 (or -1 undo) manual click.

        On +1: append an event with frame + timestamp.
        On -1: pop the most-recent event for this (line, color) cell
               so undo removes the right entry from the log."""
        if which not in LINES or color not in COLORS:
            return False
        delta = int(delta)
        with self._lock:
            new_total = max(0, self.manual_count[which][color] + delta)
            self.manual_count[which][color] = new_total
            if delta > 0:
                self._manual_seq += 1
                self.manual_events.append({
                    "seq": self._manual_seq,
                    "frame": int(frame_idx) if frame_idx is not None else None,
                    "t": time.time(),
                    "line": which,
                    "color": color,
                })
            elif delta < 0:
                # Pop the most-recent event matching (line, color).
                for i in range(len(self.manual_events) - 1, -1, -1):
                    ev = self.manual_events[i]
                    if ev["line"] == which and ev["color"] == color:
                        self.manual_events.pop(i)
                        break
            self._save()
        return True

    def reset_manual(self):
        with self._lock:
            for n in LINES:
                for c in COLORS:
                    self.manual_count[n][c] = 0
            self.manual_events = []
            self._manual_seq = 0
            self.manual_source_video = None
            self._save()

    def tag_manual_source(self, video_path):
        """Record which video the ground-truth was collected against,
        so when you load a clip later we can warn if the saved manual
        counts came from a different file."""
        with self._lock:
            if video_path != getattr(self, "manual_source_video", None):
                self.manual_source_video = video_path
                self._save()

    def calibrate_color(self, frame_bgr, color, cx, cy, r, set_size=False):
        """Sample HSV inside a circle (cx, cy, r) for the given color.
        If set_size=True, also store ball_area_px = π·r² as the reference."""
        if color not in COLORS:
            return False, {"reason": f"unknown color {color}"}
        h, w = frame_bgr.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return False, {"reason": f"({cx},{cy}) outside frame"}
        if r < 3:
            return False, {"reason": "radius too small"}

        # Mask: pixels inside the circle
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, thickness=-1)

        # Convert to HSV. Use MEDIAN of pixels inside the circle, not
        # mean — if a metal rod or stripe cuts across the ball, the rod
        # pixels are outliers and mean would drift toward gray. Median
        # is the value at the 50th percentile, so up to ~half the
        # pixels can be non-ball without affecting the result.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        inside = circle_mask > 0
        h_med = int(np.median(hsv[:, :, 0][inside]))
        s_med = int(np.median(hsv[:, :, 1][inside]))
        v_med = int(np.median(hsv[:, :, 2][inside]))

        with self._lock:
            self.hsv[color] = (h_med, s_med, v_med)
            if set_size:
                self.ball_area_px = int(round(np.pi * r * r))
            self._prev_blobs[color] = []
            self._save()

        return True, {
            "color": color,
            "hsv": [h_med, s_med, v_med],
            "radius": int(r),
            "ball_area_px": self.ball_area_px,
            "set_size": bool(set_size),
        }

    # ---------------- frame processing ----------------
    def _mask_for(self, frame_bgr, hsv_img, color):
        center = self.hsv[color]
        if center is None:
            return None
        h, s, v = center
        dh, ds, dv = self.hsv_tol
        lo = np.array([max(0, h - dh), max(0, s - ds), max(0, v - dv)],
                      dtype=np.uint8)
        hi = np.array([min(180, h + dh), min(255, s + ds), min(255, v + dv)],
                      dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lo, hi)
        # Small open kernel kills isolated single-pixel noise.
        open_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        # CLOSE with a larger kernel — this is what bridges rod gaps.
        # k = 2*r+1, so r=4 → 9×9 kernel that fills gaps up to ~9px.
        r = max(1, int(self.mask_close_radius))
        close_size = 2 * r + 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def _find_blobs(self, mask):
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        min_area = max(40, int((self.ball_area_px or 800) * 0.3))
        out = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            # Aspect-ratio filter: motion-blur trails are very tall +
            # thin (or wide + short for horizontal lines). Real ball
            # blobs are roughly square (single) or moderately rectangular
            # (cluster). Anything outside [aspect_min, aspect_max] gets
            # rejected — this is the key fix for the "f200 56×119" type
            # of spurious crossing.
            ar = w / max(1.0, float(h))
            if ar < self.aspect_min or ar > self.aspect_max:
                continue
            cx, cy = centroids[i]
            out.append({
                "cx": float(cx), "cy": float(cy), "area": area,
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": w, "h": h, "aspect": round(ar, 2),
            })
        return out

    def _check_crossings(self, color, cur_blobs):
        """Returns ({line_name: total_balls_added}, list of crossing dicts).

        Cooldown logic: after firing a crossing on a given line, we
        record (cx, cy, frame_idx). For the next cooldown_frames frames,
        any new crossing on the SAME line whose centroid is within
        (ball_diameter * cooldown_radius_mult) px of a recent crossing
        gets suppressed. Stops the same physical ball's motion-blur
        trail / shadow / mask-flicker from registering as multiple
        crossings on consecutive frames.
        """
        with self._lock:
            lines = dict(self.lines)
            ball_area = self.ball_area_px
            prev = self._prev_blobs[color]
            cooldown_frames = self.cooldown_frames
            cooldown_mult = self.cooldown_radius_mult
            recent_by_line = {n: list(self._recent_crossings[n])
                              for n in LINES}
        added = {n: 0 for n in LINES}
        events = []
        if not prev or ball_area is None:
            return added, events
        r2 = self.match_radius_px ** 2
        # Cooldown radius derived from calibrated ball diameter.
        ball_diam = max(1.0, 2.0 * (ball_area / np.pi) ** 0.5)
        cool_r2 = (ball_diam * cooldown_mult) ** 2
        new_recent = {n: [] for n in LINES}
        for cb in cur_blobs:
            best = None
            for pb in prev:
                d2 = (cb["cx"] - pb["cx"]) ** 2 + (cb["cy"] - pb["cy"]) ** 2
                if d2 > r2:
                    continue
                if best is None or d2 < best[0]:
                    best = (d2, pb)
            if best is None:
                continue
            pb = best[1]
            for line_name, line in lines.items():
                if line is None:
                    continue
                (x1, y1), (x2, y2) = line
                if not _segments_cross(pb["cx"], pb["cy"],
                                       cb["cx"], cb["cy"],
                                       x1, y1, x2, y2):
                    continue
                # COOLDOWN: skip if a recent crossing on this line was
                # near this centroid within cooldown_frames.
                cooled = False
                for (rx, ry, rf) in recent_by_line[line_name]:
                    if (self._frame_idx - rf) > cooldown_frames:
                        continue
                    if (cb["cx"] - rx) ** 2 + (cb["cy"] - ry) ** 2 <= cool_r2:
                        cooled = True
                        break
                if cooled:
                    cb["_suppressed"] = self._frame_idx
                    events.append({
                        "line": line_name, "color": color,
                        "suppressed": True,
                        "blob_area": cb["area"],
                        "ball_area": int(ball_area),
                        "area_ratio": round(cb["area"] / float(ball_area), 2),
                        "n_balls": 0,
                        "box": [cb["x"], cb["y"], cb["w"], cb["h"]],
                        "from": [int(pb["cx"]), int(pb["cy"])],
                        "to": [int(cb["cx"]), int(cb["cy"])],
                        "aspect": cb.get("aspect"),
                    })
                    continue
                ratio = cb["area"] / float(ball_area)
                n_balls = max(1, int(round(ratio)))
                added[line_name] += n_balls
                cb["_just_crossed"] = self._frame_idx
                cb["_crossed_count"] = n_balls
                cb["_crossed_line"] = line_name
                new_recent[line_name].append((cb["cx"], cb["cy"],
                                              self._frame_idx))
                events.append({
                    "line": line_name,
                    "color": color,
                    "suppressed": False,
                    "blob_area": cb["area"],
                    "ball_area": int(ball_area),
                    "area_ratio": round(ratio, 2),
                    "n_balls": n_balls,
                    "box": [cb["x"], cb["y"], cb["w"], cb["h"]],
                    "from": [int(pb["cx"]), int(pb["cy"])],
                    "to": [int(cb["cx"]), int(cb["cy"])],
                    "aspect": cb.get("aspect"),
                })
        # Commit the new recent crossings + age out old ones.
        with self._lock:
            for n in LINES:
                self._recent_crossings[n].extend(new_recent[n])
                # Drop entries older than cooldown_frames.
                self._recent_crossings[n] = [
                    e for e in self._recent_crossings[n]
                    if (self._frame_idx - e[2]) <= cooldown_frames
                ]
        return added, events

    def _render_label_mode(self, frame_bgr):
        """Bare-bones overlay for ground-truth labeling: lines + the
        live manual counts + a LABEL badge. No HSV mask, no blob
        boxes, no mention of auto-detection — this view is just for
        the human eye to count against."""
        out = frame_bgr.copy()
        # Lines
        for line_name in LINES:
            line = self.lines[line_name]
            if line is None:
                continue
            ln_color = LINE_BGR[line_name]
            cv2.line(out, line[0], line[1], ln_color, 3)
            mid_x = (line[0][0] + line[1][0]) // 2
            mid_y = (line[0][1] + line[1][1]) // 2
            cv2.putText(out, line_name.upper(), (mid_x + 6, mid_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ln_color, 2)
        # Big LABEL badge
        cv2.rectangle(out, (8, 8), (168, 44), (0, 0, 0), -1)
        cv2.putText(out, "LABEL MODE", (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        # Manual counts so user can self-verify their click rate
        y = 76
        for ln in LINES:
            ln_color = LINE_BGR[ln]
            cv2.putText(out,
                        f"{ln.upper():<11}  P={self.manual_count[ln]['purple']}"
                        f"  G={self.manual_count[ln]['green']}",
                        (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ln_color, 2)
            y += 30
        return out

    def step(self, frame_bgr, *, advance_state=True, label_mode=False):
        if advance_state:
            self._frame_idx += 1

        # LABEL MODE: skip everything HSV/blob related. Just draw the
        # lines + manual counts + a big LABEL badge so the user knows
        # they're in ground-truth-collection mode.
        if label_mode:
            return self._render_label_mode(frame_bgr)

        if not self.calibrated_any:
            out = frame_bgr.copy()
            cv2.putText(out,
                        "1. Click 'Calibrate purple' then click on a still ball.",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            return out

        hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        per_color = {}
        for color in COLORS:
            mask = self._mask_for(frame_bgr, hsv_img, color)
            zero_added = {n: 0 for n in LINES}
            if mask is None:
                per_color[color] = {"blobs": [], "added": dict(zero_added),
                                    "mask": None, "events": []}
                continue
            blobs = self._find_blobs(mask)
            if advance_state:
                added, events = self._check_crossings(color, blobs)
            else:
                added, events = dict(zero_added), []
            per_color[color] = {"blobs": blobs, "added": added,
                                "mask": mask, "events": events}

        if advance_state:
            with self._lock:
                for c in COLORS:
                    for ln in LINES:
                        self.count[ln][c] += per_color[c]["added"][ln]
                    self._prev_blobs[c] = per_color[c]["blobs"]
                    self.last_blob_count[c] = len(per_color[c]["blobs"])
                    for ev in per_color[c]["events"]:
                        self._cross_seq += 1
                        ev_full = {"seq": self._cross_seq,
                                   "frame": self._frame_idx, **ev,
                                   "count_after": self.count[ev["line"]][c]}
                        self.crossings.append(ev_full)
                if len(self.crossings) > 200:
                    self.crossings = self.crossings[-200:]

        # ---------------- annotation ----------------
        # Base canvas: full-color frame, or BLACK if mask-only mode.
        if self.view_mode == "mask":
            out = np.zeros_like(frame_bgr)
        else:
            out = frame_bgr.copy()

        # Tinted mask overlay per color
        for color in COLORS:
            mask = per_color[color]["mask"]
            if mask is None:
                continue
            tint = np.zeros_like(frame_bgr)
            tint[mask > 0] = COLOR_BGR[color]
            alpha = 0.65 if self.view_mode == "mask" else 0.30
            out = cv2.addWeighted(out, 1.0, tint, alpha, 0)

        ball_area = self.ball_area_px or 1

        # Blob boxes + DETAILED labels (area + count + diameter)
        # Box color = the calibrated color; YELLOW if this blob just crossed.
        for color in COLORS:
            box_color = COLOR_BGR[color]
            for cb in per_color[color]["blobs"]:
                area = cb["area"]
                ratio = area / float(ball_area)
                n_balls = max(1, int(round(ratio)))
                # Highlight blobs that crossed within the last few frames
                age = (self._frame_idx - cb.get("_just_crossed", -999)
                       if "_just_crossed" in cb else None)
                crossed = age is not None and age <= self._highlight_frames
                if crossed:
                    use_color = (0, 255, 255)  # yellow
                    thickness = 3
                else:
                    use_color = box_color
                    thickness = 2
                cv2.rectangle(out, (cb["x"], cb["y"]),
                              (cb["x"] + cb["w"], cb["y"] + cb["h"]),
                              use_color, thickness)
                # Detailed label: area, area-ratio, ball count
                label = f"area={area}  ratio={ratio:.2f}  n={n_balls}"
                # Black backdrop for legibility on tinted bg
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.45, 1)
                lx, ly = cb["x"], max(th + 4, cb["y"] - 4)
                cv2.rectangle(out, (lx, ly - th - 4),
                              (lx + tw + 4, ly + 2), (0, 0, 0), -1)
                cv2.putText(out, label, (lx + 2, ly - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, use_color, 1)
                # If this blob just crossed, add a +N flag inside the box
                if crossed:
                    flag = f"+{cb.get('_crossed_count', n_balls)}"
                    cv2.putText(out, flag,
                                (cb["x"] + cb["w"] // 2 - 12,
                                 cb["y"] + cb["h"] // 2 + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)

        # Counting lines (both) + reference ball-size circle next to
        # classified line endpoint as a visual yardstick.
        for line_name in LINES:
            line = self.lines[line_name]
            if line is None:
                continue
            ln_color = LINE_BGR[line_name]
            cv2.line(out, line[0], line[1], ln_color, 3)
            # Label
            mid_x = (line[0][0] + line[1][0]) // 2
            mid_y = (line[0][1] + line[1][1]) // 2
            cv2.putText(out, line_name.upper(), (mid_x + 6, mid_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ln_color, 2)
        # Reference circle = 1 calibrated ball, on the classified line endpoint
        if self.lines["classified"] is not None and self.ball_area_px:
            r_ref = max(4, int(round((self.ball_area_px / np.pi) ** 0.5)))
            p = self.lines["classified"][0]
            cv2.circle(out, p, r_ref, LINE_BGR["classified"], 2)
            cv2.putText(out, f"1 ball = r{r_ref}px",
                        (p[0] + r_ref + 4, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        LINE_BGR["classified"], 1)

        # Header counts: AUTO vs MANUAL per line, per color
        line_h = 26
        y = 36
        for line_name in LINES:
            ln_color = LINE_BGR[line_name]
            cv2.putText(out,
                        f"{line_name.upper():<10}  "
                        f"P {self.count[line_name]['purple']}"
                        f"/{self.manual_count[line_name]['purple']}  "
                        f"G {self.count[line_name]['green']}"
                        f"/{self.manual_count[line_name]['green']}",
                        (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ln_color, 2)
            y += line_h
        cv2.putText(out, "  (auto/manual)", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        # Footer: calibration info + tolerance for at-a-glance debugging
        cv2.putText(out,
                    f"ball_area={self.ball_area_px}px  "
                    f"tol(H,S,V)=({self.hsv_tol[0]},{self.hsv_tol[1]},{self.hsv_tol[2]})  "
                    f"P={self.hsv['purple']}  G={self.hsv['green']}",
                    (16, frame_bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        return out

    def get_state(self, since_cross_seq=0):
        with self._lock:
            new_crossings = [c for c in self.crossings
                             if c["seq"] > int(since_cross_seq)]
            # Per-line per-color accuracy: min/max so it's symmetric in
            # the over- and under-count cases. Returns null if both 0
            # (nothing to compare).
            accuracy = {}
            for n in LINES:
                accuracy[n] = {}
                for c in COLORS:
                    a = self.count[n][c]
                    m = self.manual_count[n][c]
                    if a == 0 and m == 0:
                        accuracy[n][c] = None
                    else:
                        accuracy[n][c] = round(min(a, m) / max(a, m), 3)
            return {
                "calibrated_any": self.calibrated_any,
                "calibrated_all": self.calibrated_all,
                "count": {n: dict(self.count[n]) for n in LINES},
                "manual_count": {n: dict(self.manual_count[n]) for n in LINES},
                "manual_events": list(self.manual_events),
                "manual_source_video": getattr(self, "manual_source_video", None),
                "accuracy": accuracy,
                "hsv": {c: list(self.hsv[c]) if self.hsv[c] else None
                        for c in COLORS},
                "hsv_tol": list(self.hsv_tol),
                "ball_area_px": self.ball_area_px,
                "lines": {n: ([list(self.lines[n][0]), list(self.lines[n][1])]
                              if self.lines[n] else None)
                          for n in LINES},
                "blob_count": dict(self.last_blob_count),
                "view_mode": self.view_mode,
                "mask_close_radius": self.mask_close_radius,
                "aspect_min": self.aspect_min,
                "aspect_max": self.aspect_max,
                "cooldown_frames": self.cooldown_frames,
                "cooldown_radius_mult": self.cooldown_radius_mult,
                "crossings": new_crossings,
                "cross_seq": self._cross_seq,
            }

    def reset_count(self):
        with self._lock:
            for n in LINES:
                for c in COLORS:
                    self.count[n][c] = 0
            for c in COLORS:
                self._prev_blobs[c] = []
            self.crossings = []
            self._cross_seq = 0
            self._frame_idx = 0
