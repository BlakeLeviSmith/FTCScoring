"""
YOLOv8 nano ball detector for FTC DECODE scoring.
Drop-in replacement for BallDetector — same detect() / draw_detections() interface.

Adds lightweight per-stream IoU tracking so balls get stable track_ids across frames.
This is used by RampTracker to count gate entries based on ID transitions rather
than raw count deltas (more robust to missed frames of detection).
"""

import cv2
import numpy as np
import config
from detector import StableDetector


def _iou(box_a, box_b):
    """Compute IoU between two boxes in (x, y, w, h) pixel format."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class SimpleIoUTracker:
    """Assigns persistent IDs to balls based on IoU overlap between frames.

    Not as sophisticated as ByteTrack, but works per-stream (each alliance ROI crop
    has its own instance), so there's no cross-contamination between alliances.

    Hysteresis: a track is kept alive for up to `max_missed` frames with no
    matching detection, so a single frame of missed detection doesn't lose the ID.
    """

    def __init__(self, iou_threshold=0.1, max_missed=60):
        self.next_id = 1
        # id -> {"bbox": (x,y,w,h), "missed": int, "color": "G"|"P"}
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def reset(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        """Assign track_ids to a list of detection dicts (mutates them in place).

        Returns the same list for convenience.
        """
        # Build matrix of IoU between existing tracks and new detections,
        # restricted to same-color matches (we never want a green ball to
        # inherit a purple ball's ID).
        track_ids = list(self.tracks.keys())
        assigned_tracks = set()
        assigned_dets = set()

        # Greedy match: strongest IoU first.
        pairs = []
        for ti, tid in enumerate(track_ids):
            tinfo = self.tracks[tid]
            for di, det in enumerate(detections):
                if det.get("color") != tinfo["color"]:
                    continue
                det_bbox = (
                    int(det.get("x", 0)),
                    int(det.get("y", 0)),
                    int(det.get("w", 0)),
                    int(det.get("h", 0)),
                )
                score = _iou(tinfo["bbox"], det_bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di, det_bbox))

        pairs.sort(reverse=True, key=lambda p: p[0])
        for score, tid, di, det_bbox in pairs:
            if tid in assigned_tracks or di in assigned_dets:
                continue
            assigned_tracks.add(tid)
            assigned_dets.add(di)
            self.tracks[tid]["bbox"] = det_bbox
            self.tracks[tid]["missed"] = 0
            detections[di]["track_id"] = tid

        # Unmatched detections get new IDs
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            new_id = self.next_id
            self.next_id += 1
            det_bbox = (
                int(det.get("x", 0)),
                int(det.get("y", 0)),
                int(det.get("w", 0)),
                int(det.get("h", 0)),
            )
            self.tracks[new_id] = {
                "bbox": det_bbox,
                "missed": 0,
                "color": det.get("color", ""),
            }
            det["track_id"] = new_id

        # Age unmatched tracks, drop stale ones
        to_delete = []
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return detections


class YOLODetector:
    """Detects purple and green balls using a trained YOLOv8 nano model."""

    # Class indices in the training dataset
    CLASS_MAP = {0: "G", 1: "P"}
    CLASS_COLORS = {
        "G": config.COLOR_GREEN_DISPLAY,
        "P": config.COLOR_PURPLE_DISPLAY,
    }

    def __init__(self, model_path=None, confidence=None, iou_threshold=None,
                 tracking_enabled=True):
        from ultralytics import YOLO

        self._model_path = model_path or config.YOLO_MODEL_PATH
        # Primary instance is still around for `draw_detections` and anything
        # that wants a non-tracked prediction.
        self.model = YOLO(self._model_path)
        self.confidence = confidence or config.YOLO_CONFIDENCE
        self.iou_threshold = iou_threshold or config.YOLO_IOU_THRESHOLD
        self.stable = StableDetector()

        # Ultralytics' `model.track(persist=True)` stores tracker state on
        # the model's predictor. If we pass red and blue crops through the
        # same model, their tracker states interleave and corrupt each
        # other. Solution: one YOLO instance per stream_id. YOLOv8n is ~6MB
        # so two extra copies cost nothing. Each instance runs BoT-SORT,
        # which provides Kalman motion prediction + appearance ReID —
        # meaningfully more stable than the old SimpleIoUTracker,
        # especially for stationary balls being briefly occluded by robots.
        self.tracking_enabled = tracking_enabled
        self._stream_models = {}
        # Fallback IoU trackers (used only if tracking is disabled).
        self._trackers = {}

    def _get_stream_model(self, stream_id):
        from ultralytics import YOLO
        if stream_id not in self._stream_models:
            self._stream_models[stream_id] = YOLO(self._model_path)
        return self._stream_models[stream_id]

    def _get_tracker(self, stream_id):
        if stream_id not in self._trackers:
            self._trackers[stream_id] = SimpleIoUTracker()
        return self._trackers[stream_id]

    def reset_tracker(self, stream_id=None):
        """Reset tracker state (e.g. on match reset or stream switch).

        For BoT-SORT we drop the model instance so the next detect() rebuilds
        with a fresh predictor/tracker. For the IoU fallback we reset the
        per-stream tracker dict.
        """
        if stream_id is None:
            self._stream_models = {}
            for t in self._trackers.values():
                t.reset()
        else:
            self._stream_models.pop(stream_id, None)
            if stream_id in self._trackers:
                self._trackers[stream_id].reset()

    def detect(self, frame, stream_id="default"):
        """
        Detect all green and purple balls in a frame using YOLOv8.

        Args:
            frame: BGR image.
            stream_id: identifier for tracker state (use a distinct id per
                alliance crop so IDs don't cross-contaminate).

        Returns:
            balls: list of dicts with keys: color, x, y, w, h, center_x, center_y,
                area, confidence, track_id (int or None).
            stable_pattern: temporally-smoothed pattern string
            raw_pattern: this frame's raw pattern string
            masks: empty dict (YOLO doesn't produce color masks)
        """
        if self.tracking_enabled:
            model = self._get_stream_model(stream_id)
            results = model.track(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                tracker="botsort.yaml",
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False,
            )

        balls = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            ids_tensor = getattr(boxes, "id", None)

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                color = self.CLASS_MAP.get(cls_id)
                if color is None:
                    continue

                # xyxy format -> x, y, w, h
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                h = y2 - y1

                tid = None
                if ids_tensor is not None:
                    tid = int(ids_tensor[i].item())

                balls.append({
                    "color": color,
                    "x": x1, "y": y1, "w": w, "h": h,
                    "center_x": x1 + w // 2,
                    "center_y": y1 + h // 2,
                    "area": w * h,
                    "confidence": conf,
                    "track_id": tid,
                })

        # If BoT-SORT is off (tracking_enabled=False), fall back to the
        # simple IoU tracker so downstream code still gets track_ids.
        if not self.tracking_enabled:
            tracker = self._get_tracker(stream_id)
            tracker.update(balls)

        # Sort left to right (same as BallDetector)
        balls.sort(key=lambda b: b["center_x"])
        raw_pattern = "".join(b["color"] for b in balls)
        stable_pattern = self.stable.update(raw_pattern)

        return balls, stable_pattern, raw_pattern, {}

    def draw_detections(self, frame, balls, stable_pattern="", raw_pattern=""):
        """Draw YOLO detection overlays on a frame copy and return it."""
        output = frame.copy()
        green_count = 0
        purple_count = 0

        for ball in balls:
            if ball["color"] == "G":
                green_count += 1
                dc = self.CLASS_COLORS["G"]
                label = f"G{green_count}"
            else:
                purple_count += 1
                dc = self.CLASS_COLORS["P"]
                label = f"P{purple_count}"

            # Draw bounding box
            x, y, w, h = ball["x"], ball["y"], ball["w"], ball["h"]
            cv2.rectangle(output, (x, y), (x + w, y + h), dc, 2)

            # Label with confidence and (if present) track id
            conf = ball.get("confidence", 0)
            tid = ball.get("track_id")
            tid_str = f" #{tid}" if tid is not None else ""
            text = f"{label}{tid_str} {conf:.0%}" if conf else f"{label}{tid_str}"
            cv2.putText(output, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dc, 2)

        # Status overlay
        cv2.putText(output, f"[YOLO] Stable: {stable_pattern} ({len(stable_pattern)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_MATCH, 2)
        cv2.putText(output, f"Raw: {raw_pattern}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(output, f"G:{green_count} P:{purple_count}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1)

        return output
