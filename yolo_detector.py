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


def _boost_green(frame):
    """Selectively increase saturation+brightness in the green hue range.

    Boosts S and V only for pixels where hue is in the green range (35-85).
    Other pixels (purple, ramp, robots, etc.) are left untouched.

    NOTE: cv2.add(src, scalar, mask=...) ZEROS OUT non-masked pixels — that
    was a bug in an earlier version that killed all detection. We use direct
    numpy indexing instead so non-green pixels stay exactly as they were.
    """
    sat_boost = config.GREEN_SAT_BOOST
    val_boost = config.GREEN_VAL_BOOST
    if sat_boost == 0 and val_boost == 0:
        return frame  # boost disabled, return as-is
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green_idx = (h >= 35) & (h <= 85)
    if sat_boost:
        s_new = s.astype(np.int16)
        s_new[green_idx] = np.clip(s_new[green_idx] + sat_boost, 0, 255)
        s = s_new.astype(np.uint8)
    if val_boost:
        v_new = v.astype(np.int16)
        v_new[green_idx] = np.clip(v_new[green_idx] + val_boost, 0, 255)
        v = v_new.astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


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

    def swap_model(self, new_path):
        """Hot-swap the YOLO weights without restarting the app.

        Reloads the primary model and drops every per-stream model so the
        next detect() call picks up the new weights everywhere. Tracker
        state is reset as a side effect (BoT-SORT can't carry across a
        model change). Returns the resolved path actually loaded.
        """
        from ultralytics import YOLO
        self.model = YOLO(new_path)
        self._model_path = new_path
        self._stream_models = {}
        return new_path

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
        # Pre-process: boost green saturation so lighter green balls
        # produce higher confidence detections from YOLO.
        enhanced = _boost_green(frame)

        tracker_cfg = getattr(config, "YOLO_TRACKER_CONFIG", "botsort.yaml")
        use_tta = bool(getattr(config, "YOLO_TTA", False))

        # Native sampling: pick imgsz to match the input frame's longest
        # side, rounded up to the nearest multiple of 32 (YOLO's stride),
        # capped by config.YOLO_MAX_IMGSZ. This way ultralytics letterboxes
        # to a square that preserves all the input pixels (no internal
        # downscale wasting HD detail). Cost scales with imgsz².
        h_in, w_in = enhanced.shape[:2]
        target = ((max(h_in, w_in) + 31) // 32) * 32
        cap = int(getattr(config, "YOLO_MAX_IMGSZ", 1280))
        imgsz = max(320, min(target, cap))

        if self.tracking_enabled:
            model = self._get_stream_model(stream_id)
            results = model.track(
                enhanced,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=imgsz,
                tracker=tracker_cfg,
                persist=True,
                augment=use_tta,
                verbose=False,
            )
        else:
            results = self.model.predict(
                enhanced,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=imgsz,
                augment=use_tta,
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

        # Per-class confidence filter: green gets a lower bar than purple
        # because the model is less confident on lighter-colored balls.
        per_class = getattr(config, 'YOLO_CONF_PER_CLASS', None)
        if per_class:
            balls = [b for b in balls
                     if b["confidence"] >= per_class.get(b["color"], 0)]

        # Position-dependent confidence: far end of the ROI (top of crop,
        # near the gate) has smaller balls → accept lower confidence there.
        # Near end (bottom) stays strict to avoid false positives.
        far_frac = getattr(config, 'YOLO_FAR_REGION_FRACTION', 0)
        if far_frac > 0:
            h_frame = frame.shape[0]
            conf_far = getattr(config, 'YOLO_CONF_FAR', self.confidence)
            conf_near = getattr(config, 'YOLO_CONF_NEAR', self.confidence)
            filtered = []
            for b in balls:
                y_frac = b["center_y"] / h_frame
                threshold = conf_far if y_frac < far_frac else conf_near
                if b["confidence"] >= threshold:
                    filtered.append(b)
            balls = filtered

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
