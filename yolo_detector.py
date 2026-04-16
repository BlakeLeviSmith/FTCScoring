"""
YOLOv8 nano ball detector for FTC DECODE scoring.

Uses ultralytics' built-in BoT-SORT tracker (model.track with persist=True)
for per-stream ball tracking. Each alliance ROI gets its own model instance
to keep tracker state isolated.
"""

import cv2
import numpy as np
import config
from detector import StableDetector


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

    def _get_stream_model(self, stream_id):
        from ultralytics import YOLO
        if stream_id not in self._stream_models:
            self._stream_models[stream_id] = YOLO(self._model_path)
        return self._stream_models[stream_id]

    def reset_tracker(self, stream_id=None):
        """Reset tracker state by dropping per-stream model instances.

        The next detect() call will rebuild a fresh model + BoT-SORT tracker.
        """
        if stream_id is None:
            self._stream_models = {}
        else:
            self._stream_models.pop(stream_id, None)

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

        # Sort left to right
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
