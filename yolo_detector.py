"""
YOLOv8 nano ball detector for FTC DECODE scoring.
Drop-in replacement for BallDetector — same detect() / draw_detections() interface.
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

    def __init__(self, model_path=None, confidence=None, iou_threshold=None):
        from ultralytics import YOLO

        path = model_path or config.YOLO_MODEL_PATH
        self.model = YOLO(path)
        self.confidence = confidence or config.YOLO_CONFIDENCE
        self.iou_threshold = iou_threshold or config.YOLO_IOU_THRESHOLD
        self.stable = StableDetector()

    def detect(self, frame):
        """
        Detect all green and purple balls in a frame using YOLOv8.

        Returns:
            balls: list of dicts with keys: color, x, y, w, h, center_x, center_y, area, confidence
            stable_pattern: temporally-smoothed pattern string
            raw_pattern: this frame's raw pattern string
            masks: empty dict (YOLO doesn't produce color masks)
        """
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

                balls.append({
                    "color": color,
                    "x": x1, "y": y1, "w": w, "h": h,
                    "center_x": x1 + w // 2,
                    "center_y": y1 + h // 2,
                    "area": w * h,
                    "confidence": conf,
                })

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

            # Label with confidence
            conf = ball.get("confidence", 0)
            text = f"{label} {conf:.0%}" if conf else label
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
