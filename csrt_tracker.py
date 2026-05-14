"""
MultiBallTracker — per-ball CSRT correlation-filter tracking.

This replaces ByteTrack's "track-by-detection" pattern (which relies on
YOLO firing a clean detection EVERY frame) with appearance-based
tracking: each ball gets its own OpenCV CSRT correlation filter that
follows its appearance through pixels frame-to-frame, independent of
detection consistency.

Why this works for fast/unpredictable motion:
  - CSRT scans a search window around the ball's last known position
    on each frame and locks onto the strongest appearance match.
  - It doesn't extrapolate motion (no Kalman) — it just chases
    appearance — so abrupt direction reversals (bounces) are handled
    natively.
  - YOLO is used only as an ANCHOR: when YOLO sees a ball clearly,
    we re-initialize / refresh the matching tracker to correct drift.

Per-frame loop (call `step()`):
  1. Update every existing CSRT tracker — ok or lost
  2. Match new YOLO detections to existing trackers by IoU
     - matched: refresh tracker bbox + color + confidence (anchor)
     - unmatched detection: spawn a new tracker
     - unmatched alive tracker: keep it running on appearance alone
  3. Retire trackers that:
     - return ok=False from CSRT's internal check
     - have drifted off the masked region (no pixels = no signal)
     - have been alive without any YOLO anchor for too long (drift risk)
  4. Return list of {tid, x, y, w, h, center_x, center_y, color,
                     confidence} for all alive trackers this frame

Designed for ~5-20 concurrent balls. CSRT runs at hundreds of fps per
tracker on CPU, but with 20 trackers @ 30fps that's 600 tracker
updates/sec — still trivial on a modern Mac.
"""

import time
from collections import deque

import cv2
import numpy as np


def _estimate_trail_velocity(trail):
    """Estimate (vx, vy) in px/frame from the last few trail points.
    Used to coast a CSRT bbox forward during ok=False frames so the
    tracker keeps moving along the ball's trajectory during occlusions
    or motion-blur drops (gate-to-ramp). Returns (0, 0) if the trail
    is too short to estimate."""
    if not trail or len(trail) < 2:
        return 0.0, 0.0
    recent = list(trail)[-6:]
    x0, y0, f0, _ = recent[0]
    x1, y1, f1, _ = recent[-1]
    df = max(1, f1 - f0)
    return (x1 - x0) / df, (y1 - y0) / df


def _iou(a, b):
    """IoU of two boxes in (x, y, w, h) format."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class _Track:
    """One per-ball state holder around an OpenCV CSRT tracker."""

    __slots__ = ("tid", "tracker", "bbox", "color", "confidence",
                 "first_seen", "last_seen_yolo", "last_alive_frame",
                 "lost_frames", "trail")

    def __init__(self, tid, tracker, bbox, color, confidence,
                 frame_idx, trail_length=30):
        self.tid = tid
        self.tracker = tracker
        self.bbox = bbox  # (x, y, w, h) — current best estimate
        self.color = color
        self.confidence = confidence
        self.first_seen = frame_idx
        self.last_seen_yolo = frame_idx
        self.last_alive_frame = frame_idx
        self.lost_frames = 0
        self.trail = deque(maxlen=trail_length)
        # Seed trail with the initial position
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.trail.append((float(cx), float(cy), int(frame_idx), str(color or "?")))

    def to_dict(self):
        cx = self.bbox[0] + self.bbox[2] / 2.0
        cy = self.bbox[1] + self.bbox[3] / 2.0
        return {
            "track_id": self.tid,
            "x": int(self.bbox[0]),
            "y": int(self.bbox[1]),
            "w": int(self.bbox[2]),
            "h": int(self.bbox[3]),
            "center_x": float(cx),
            "center_y": float(cy),
            "color": self.color,
            "confidence": self.confidence,
        }


class MultiBallTracker:
    """Per-alliance multi-ball CSRT tracker."""

    def __init__(self, max_lost_frames=30, match_iou=0.20,
                 max_frames_without_yolo=60, trail_length=30,
                 ghost_max_frames=45, ghost_match_radius_px=40,
                 ghost_require_color=True, max_active_tracks=12,
                 match_center_px=35, dedup_radius_px=25,
                 # Accepted for backward-compat with the brief "YOLO
                 # spots, CSRT follows" experiment; not used here.
                 match_radius_px=None, drift_reinit_px=None):
        """
        max_lost_frames:
            Frames CSRT can return ok=False before retire.
        match_iou:
            IoU threshold for matching a YOLO detection to an existing
            active tracker. 0.20 is generous; fast balls displace a lot.
        max_frames_without_yolo:
            Even if CSRT stays "ok", retire a tracker that hasn't seen
            a confirming YOLO detection in this many frames.
        trail_length:
            Per-tracker position history depth for the debug polyline.
        ghost_max_frames / ghost_match_radius_px / ghost_require_color:
            Ghost-track resurrection. When a tracker is retired we
            push (last_pos, color, original_tid) to _ghosts. If a new
            unmatched YOLO detection lands within ghost_match_radius_px
            of a ghost (and color matches if required) within
            ghost_max_frames, we RESURRECT the tracker under the
            ORIGINAL tid instead of spawning a new one. Net effect: a
            CSRT loss/re-acquire cycle keeps the same track_id, so
            the downstream tripwire counter doesn't double-count.
        """
        self.max_lost_frames = max_lost_frames
        self.match_iou = match_iou
        self.max_frames_without_yolo = max_frames_without_yolo
        self.trail_length = trail_length
        self.ghost_max_frames = ghost_max_frames
        self.ghost_match_radius_px = ghost_match_radius_px
        self.ghost_require_color = ghost_require_color
        self.max_active_tracks = int(max_active_tracks)
        self.match_center_px = int(match_center_px)
        # Same-color trackers within this many pixels of each other are
        # considered duplicates of the same physical ball. The younger
        # one is retired (silently — no ghost) so it doesn't get re-
        # spawned next frame.
        self.dedup_radius_px = int(dedup_radius_px)
        self._tracks = {}   # tid -> _Track
        self._ghosts = []   # list of {tid, pos:(x,y), color, retired_frame}
        self._next_id = 1
        self._frame_idx = 0
        # Ring buffer of recent tracker events for the dashboard activity
        # log. Lets the user SEE the lifecycle of each tid (birth → coast
        # → retire → resurrect → anchor) so they can diagnose why a
        # specific physical ball got dropped — e.g. "middle ball #7 was
        # retired at f1234 and never resurrected, while #6 (outer) got
        # double-anchored". Bounded so it doesn't grow without limit.
        self._events = []
        self._event_seq = 0

    def reset(self):
        self._tracks = {}
        self._ghosts = []
        self._next_id = 1
        self._frame_idx = 0
        self._events = []
        self._event_seq = 0

    def _log(self, kind, tid, **fields):
        """Append a structured event to the ring buffer."""
        self._event_seq += 1
        ev = {"seq": self._event_seq, "frame": self._frame_idx,
              "kind": kind, "tid": int(tid)}
        ev.update(fields)
        self._events.append(ev)
        if len(self._events) > 1500:
            self._events.pop(0)

    def get_events_since(self, since_seq=0):
        return [e for e in self._events if e["seq"] > since_seq]

    def _retire_track(self, tid):
        """Move a track to the ghost list and remove it from active.

        Records the ball's recent velocity so the ghost search can
        PROJECT the expected position forward each frame — this is
        what catches the gate-drop case (ball falls past zone while
        undetected, reappears below; the projected ghost lands near
        the new detection so the original tid is preserved)."""
        if tid not in self._tracks:
            return
        tr = self._tracks.pop(tid)
        cx = tr.bbox[0] + tr.bbox[2] / 2.0
        cy = tr.bbox[1] + tr.bbox[3] / 2.0
        # Estimate velocity (px/frame) from the last few trail points.
        vx, vy = 0.0, 0.0
        if len(tr.trail) >= 2:
            recent = list(tr.trail)[-6:]
            x0, y0, f0, _ = recent[0]
            x1, y1, f1, _ = recent[-1]
            df = max(1, f1 - f0)
            vx = (x1 - x0) / df
            vy = (y1 - y0) / df
        self._ghosts.append({
            "tid": tid,
            "pos": (float(cx), float(cy)),
            "vel": (float(vx), float(vy)),
            "color": tr.color,
            "retired_frame": self._frame_idx,
        })
        self._log("retire", tid, color=tr.color,
                  pos=[int(cx), int(cy)],
                  vel=[round(vx, 1), round(vy, 1)],
                  lost_frames=int(tr.lost_frames))

    def _find_matching_ghost(self, x, y, color):
        """Return (index, ghost_dict) for the closest ghost within
        ghost_match_radius_px and color match, else None.

        Each ghost's stored position is PROJECTED forward by its
        velocity × frames-since-retirement before computing distance,
        so a ball that was last seen above the gate and falls 100 px
        in 5 frames will be matched against a projected position near
        the new ramp detection."""
        r2 = self.ghost_match_radius_px ** 2
        best = None  # (dist², idx)
        for i, g in enumerate(self._ghosts):
            if self.ghost_require_color and g.get("color") and color \
                    and g["color"] != color:
                continue
            df = self._frame_idx - g["retired_frame"]
            gx0, gy0 = g["pos"]
            vx, vy = g.get("vel", (0.0, 0.0))
            # Clamp projection to a sane horizon so a noisy velocity
            # estimate doesn't fling the ghost off-frame.
            df_clamp = min(df, self.ghost_max_frames)
            gx = gx0 + vx * df_clamp
            gy = gy0 + vy * df_clamp
            d2 = (x - gx) ** 2 + (y - gy) ** 2
            if d2 > r2:
                continue
            if best is None or d2 < best[0]:
                best = (d2, i)
        if best is None:
            return None
        return best[1], self._ghosts[best[1]]

    def _evict_stale_ghosts(self):
        cutoff = self._frame_idx - self.ghost_max_frames
        self._ghosts = [g for g in self._ghosts if g["retired_frame"] >= cutoff]

    def step(self, frame_bgr, yolo_detections):
        """One frame update.

        Args:
            frame_bgr: BGR image the trackers operate on. Should be the
                same coord space as yolo_detections' x/y/w/h.
            yolo_detections: list of dicts with x, y, w, h, color,
                confidence (in frame_bgr coords).

        Returns:
            List of dicts (track_id, x, y, w, h, center_x, center_y,
            color, confidence) — one per alive tracker this frame.
        """
        self._frame_idx += 1

        # 1. Update every existing CSRT tracker on the new frame.
        # When CSRT returns ok=False (the gate-to-ramp drop is the
        # canonical case — motion blur during fall, CSRT can't lock),
        # we DON'T freeze the bbox. Instead we advance it by the ball's
        # recent velocity so the tracker keeps moving along the
        # expected trajectory through the blind frames. When YOLO
        # re-acquires the ball on the ramp below, the coasting bbox
        # is near the new detection and the IoU/center match in step 2
        # succeeds — same tid preserved across the drop.
        to_retire = []
        for tid, tr in self._tracks.items():
            try:
                ok, bbox = tr.tracker.update(frame_bgr)
            except cv2.error:
                ok, bbox = False, tr.bbox
            if ok:
                tr.bbox = tuple(map(int, bbox))
                tr.last_alive_frame = self._frame_idx
                tr.lost_frames = 0
            else:
                tr.lost_frames += 1
                # COAST: dead-reckon the bbox using last known velocity.
                # Estimated from the last few trail points (px/frame).
                vx, vy = _estimate_trail_velocity(tr.trail)
                # Bias slightly toward gravity if velocity estimate is
                # near zero — the drop case has near-zero velocity in
                # the frames right before the ball leaves the gate.
                if abs(vx) < 0.5 and abs(vy) < 0.5:
                    vy = 4.0  # small downward nudge (px/frame)
                x, y, w_, h_ = tr.bbox
                tr.bbox = (int(x + vx), int(y + vy), int(w_), int(h_))
                if tr.lost_frames == 1:
                    # Log only the FIRST lost frame; "coasting" is a
                    # state, not a per-frame event.
                    self._log("coast_start", tid, color=tr.color,
                              vel=[round(vx, 1), round(vy, 1)],
                              pos=[int(x + w_ / 2), int(y + h_ / 2)])
                if tr.lost_frames > self.max_lost_frames:
                    to_retire.append(tid)
            # Drift guard: too long without a YOLO anchor
            if (self._frame_idx - tr.last_seen_yolo) > self.max_frames_without_yolo:
                to_retire.append(tid)

        for tid in set(to_retire):
            # Move retired track to the ghost list for potential
            # resurrection if a new YOLO detection appears at its
            # last known position within ghost_max_frames.
            self._retire_track(tid)

        # 2. Match YOLO detections to existing trackers by IoU (greedy).
        #
        # COLOR-AWARE: a same-color candidate is ALWAYS preferred over a
        # cross-color candidate, regardless of which has higher IoU. This
        # is the fix for the cluster-drop tid swap where a green YOLO
        # box was claiming a purple tracker (or vice versa) just because
        # the bounding boxes happened to overlap during the fall. We
        # encode this by using a tuple key (-color_match, iou) so the
        # max() prefers same-color matches, then the higher IoU.
        det_boxes = [(d.get("x", 0), d.get("y", 0),
                      d.get("w", 0), d.get("h", 0)) for d in yolo_detections]
        unmatched_dets = list(range(len(det_boxes)))
        claimed = set()
        while unmatched_dets:
            best = None  # (color_match_int, iou, det_idx, tid)
            for di in unmatched_dets:
                db = det_boxes[di]
                d_color = yolo_detections[di].get("color")
                for tid, tr in self._tracks.items():
                    if tid in claimed:
                        continue
                    iou = _iou(db, tr.bbox)
                    if iou < self.match_iou:
                        continue
                    color_match = 1 if (d_color and tr.color
                                        and d_color == tr.color) else 0
                    cand = (color_match, iou, di, tid)
                    if best is None or cand > best:
                        best = cand
            if best is None:
                break
            cm, _iou_val, di, tid = best
            tr = self._tracks[tid]
            db = det_boxes[di]
            # Anchor: reinitialize CSRT with the YOLO box (corrects drift)
            new_tracker = self._make_csrt()
            try:
                new_tracker.init(frame_bgr, db)
                tr.tracker = new_tracker
                tr.bbox = db
                tr.lost_frames = 0
                tr.last_alive_frame = self._frame_idx
                tr.last_seen_yolo = self._frame_idx
                d = yolo_detections[di]
                tr.color = d.get("color") or tr.color
                tr.confidence = d.get("confidence", tr.confidence)
            except cv2.error:
                pass
            self._log("anchor_iou", tid, color=tr.color,
                      pos=[int(db[0] + db[2] / 2), int(db[1] + db[3] / 2)],
                      iou=round(_iou_val, 2),
                      cross_color=0 if cm else 1)
            claimed.add(tid)
            unmatched_dets.remove(di)

        # 2b. SECOND-PASS center-distance match. Same color-aware logic:
        # prefer same-color trackers; only allow cross-color if no
        # same-color candidate is in range.
        r2 = self.match_center_px ** 2
        still_unmatched = []
        for di in unmatched_dets:
            db = det_boxes[di]
            dcx = db[0] + db[2] / 2.0
            dcy = db[1] + db[3] / 2.0
            d_color = yolo_detections[di].get("color")
            best = None  # (color_match_int_neg, dist², tid)
            for tid, tr in self._tracks.items():
                if tid in claimed:
                    continue
                tcx = tr.bbox[0] + tr.bbox[2] / 2.0
                tcy = tr.bbox[1] + tr.bbox[3] / 2.0
                d2 = (dcx - tcx) ** 2 + (dcy - tcy) ** 2
                if d2 > r2:
                    continue
                color_match = 1 if (d_color and tr.color
                                    and d_color == tr.color) else 0
                # Same-color first (color_match=1 > 0), then nearer
                # (smaller d² → use -d² so max picks smallest).
                cand = (color_match, -d2, tid)
                if best is None or cand > best:
                    best = cand
            if best is None:
                still_unmatched.append(di)
                continue
            cm, neg_d2, tid = best
            tr = self._tracks[tid]
            new_tracker = self._make_csrt()
            try:
                new_tracker.init(frame_bgr, db)
                tr.tracker = new_tracker
                tr.bbox = db
                tr.lost_frames = 0
                tr.last_alive_frame = self._frame_idx
                tr.last_seen_yolo = self._frame_idx
                d = yolo_detections[di]
                tr.color = d.get("color") or tr.color
                tr.confidence = d.get("confidence", tr.confidence)
            except cv2.error:
                pass
            self._log("anchor_dist", tid, color=tr.color,
                      pos=[int(dcx), int(dcy)],
                      dist=int((-neg_d2) ** 0.5),
                      cross_color=0 if cm else 1)
            claimed.add(tid)
        unmatched_dets = still_unmatched

        # 3. Unmatched YOLO detections — either RESURRECT a matching
        # ghost (same physical ball CSRT just lost) or spawn fresh CSRT.
        for di in unmatched_dets:
            db = det_boxes[di]
            d = yolo_detections[di]
            cx = db[0] + db[2] / 2.0
            cy = db[1] + db[3] / 2.0
            color = d.get("color")
            # Hard cap on active trackers — CSRT.update() runs per-tracker
            # per-frame and burns ~20-50ms each. Without a cap, runaway
            # false-positive spawning collapses processing FPS.
            ghost_match = self._find_matching_ghost(cx, cy, color)
            if ghost_match is None and len(self._tracks) >= self.max_active_tracks:
                continue
            tracker = self._make_csrt()
            try:
                tracker.init(frame_bgr, db)
            except cv2.error:
                continue
            # Check ghosts first — same physical ball, preserve tid.
            if ghost_match is not None:
                gi, ghost = ghost_match
                tid = ghost["tid"]
                self._ghosts.pop(gi)
                self._tracks[tid] = _Track(
                    tid=tid,
                    tracker=tracker,
                    bbox=db,
                    color=color or ghost.get("color"),
                    confidence=d.get("confidence", 0.0),
                    frame_idx=self._frame_idx,
                    trail_length=self.trail_length,
                )
                # Distance between projected-ghost and actual detection
                # is the most diagnostic number for the drop case.
                df = self._frame_idx - ghost["retired_frame"]
                vx, vy = ghost.get("vel", (0.0, 0.0))
                gx0, gy0 = ghost["pos"]
                df_clamp = min(df, self.ghost_max_frames)
                gx = gx0 + vx * df_clamp
                gy = gy0 + vy * df_clamp
                proj_err = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                self._log("resurrect", tid, color=color,
                          pos=[int(cx), int(cy)],
                          proj=[int(gx), int(gy)],
                          proj_err=int(proj_err),
                          gap_frames=int(df))
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(
                    tid=tid,
                    tracker=tracker,
                    bbox=db,
                    color=color,
                    confidence=d.get("confidence", 0.0),
                    frame_idx=self._frame_idx,
                    trail_length=self.trail_length,
                )
                self._log("birth", tid, color=color,
                          pos=[int(cx), int(cy)])

        # 3b. DEDUPLICATE near-coincident trackers of the same color.
        # The raised NMS IoU lets touching real balls survive (good)
        # but ALSO lets two boxes on the SAME ball survive — those each
        # spawn their own tracker, producing 4 trackers on 2 actual
        # balls. After matching, walk the active set: any pair of
        # same-color trackers whose centers are within dedup_radius_px
        # gets merged by retiring the YOUNGER (or lower-confidence) one.
        # This prevents the tripwire counter from registering both as
        # separate balls, which is the immediate cause of the stagnant-
        # ball overcount in your debug trace.
        dedup_r2 = float(self.dedup_radius_px) ** 2
        tids = list(self._tracks.keys())
        retired_dup = set()
        for i in range(len(tids)):
            tid_a = tids[i]
            if tid_a in retired_dup:
                continue
            tr_a = self._tracks.get(tid_a)
            if tr_a is None:
                continue
            ax = tr_a.bbox[0] + tr_a.bbox[2] / 2.0
            ay = tr_a.bbox[1] + tr_a.bbox[3] / 2.0
            for j in range(i + 1, len(tids)):
                tid_b = tids[j]
                if tid_b in retired_dup:
                    continue
                tr_b = self._tracks.get(tid_b)
                if tr_b is None:
                    continue
                if tr_a.color != tr_b.color:
                    continue
                bx = tr_b.bbox[0] + tr_b.bbox[2] / 2.0
                by = tr_b.bbox[1] + tr_b.bbox[3] / 2.0
                d2 = (ax - bx) ** 2 + (ay - by) ** 2
                if d2 > dedup_r2:
                    continue
                # Same color, within dedup radius — duplicate.
                # Keep the older (lower tid number = born first); the
                # younger one is the spurious duplicate that needs to die.
                # Don't push to ghosts (would just resurrect again).
                victim = tid_b if tid_a < tid_b else tid_a
                survivor = tid_a if tid_a < tid_b else tid_b
                retired_dup.add(victim)
                self._log("dedup", victim,
                          color=tr_a.color,
                          merged_into=int(survivor),
                          dist=int(d2 ** 0.5))
        for tid in retired_dup:
            self._tracks.pop(tid, None)

        # 4. Update trail history for all alive trackers (post-update bbox)
        for tr in self._tracks.values():
            cx = tr.bbox[0] + tr.bbox[2] / 2.0
            cy = tr.bbox[1] + tr.bbox[3] / 2.0
            tr.trail.append((float(cx), float(cy),
                             int(self._frame_idx),
                             str(tr.color or "?")))

        # 5. Evict stale ghosts that weren't resurrected in time
        self._evict_stale_ghosts()

        # 6. Per-frame summary event — single log line per frame that
        # captures what YOLO actually fired vs how many trackers we
        # ended up with. Indispensable for the "did YOLO see the ball?"
        # question. With this in the activity log you can scroll back
        # and see e.g. `f652 frame_summary yolo_dets=2(P:2 G:0)
        # active=7` — only 2 dets while 3 balls were visually present
        # → YOLO miss, not a tracker bug.
        n_yolo_p = sum(1 for d in yolo_detections if d.get("color") == "P")
        n_yolo_g = sum(1 for d in yolo_detections if d.get("color") == "G")
        active_p = sum(1 for tr in self._tracks.values() if tr.color == "P")
        active_g = sum(1 for tr in self._tracks.values() if tr.color == "G")
        # Use tid=0 because the summary isn't tied to any specific tid;
        # the JS renderer will render unknown kinds with a generic line.
        self._log("frame_summary", 0,
                  yolo_p=n_yolo_p, yolo_g=n_yolo_g,
                  active_p=active_p, active_g=active_g,
                  ghosts=len(self._ghosts))

        # 7. Return current alive trackers as ball dicts
        return [tr.to_dict() for tr in self._tracks.values()]

    def get_debug_state(self):
        """Return per-tracker state for the dashboard debug panel:
        active trackers with positions and ages, plus current ghosts."""
        cur = self._frame_idx
        active = []
        for tid, tr in self._tracks.items():
            cx = tr.bbox[0] + tr.bbox[2] / 2.0
            cy = tr.bbox[1] + tr.bbox[3] / 2.0
            active.append({
                "tid": int(tid),
                "color": tr.color,
                "pos": [int(cx), int(cy)],
                "age": int(cur - tr.first_seen),
                "lost": int(tr.lost_frames),
                "since_yolo": int(cur - tr.last_seen_yolo),
            })
        ghosts = [{
            "tid": int(g["tid"]),
            "color": g.get("color"),
            "pos": [int(g["pos"][0]), int(g["pos"][1])],
            "retired_frames_ago": int(cur - g["retired_frame"]),
        } for g in self._ghosts]
        return {
            "frame_idx": cur,
            "active": active,
            "ghosts": ghosts,
            "active_count": len(active),
            "ghost_count": len(ghosts),
            "next_id": int(self._next_id),
        }

    def get_trails(self):
        """Per-tracker trail history for debug polyline visualization."""
        return {tr.tid: list(tr.trail) for tr in self._tracks.values()}

    @staticmethod
    def _make_csrt():
        # cv2.TrackerCSRT.create() works in opencv-contrib >= 4.5.
        # If not available, fall back to KCF (faster, less accurate).
        if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
            return cv2.TrackerCSRT.create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT"):
            return cv2.legacy.TrackerCSRT.create()
        return cv2.TrackerKCF.create()
