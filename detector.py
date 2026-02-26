"""
Ball detection pipeline for FTC DECODE scoring.
Detects purple and green ARTIFACT balls using dual-mask color detection.
"""

import cv2
import numpy as np
from collections import deque
import config


class StableDetector:
    """Temporal smoothing: tracks last N frames for stable detection output."""

    def __init__(self, history_size=None, stability_threshold=None, lock_duration=None):
        hs = history_size or config.STABLE_HISTORY_SIZE
        self.history = deque(maxlen=hs)
        self.stability_threshold = stability_threshold or config.STABLE_THRESHOLD
        self.locked_motif = ""
        self.lock_frames = 0
        self.LOCK_DURATION = lock_duration or config.STABLE_LOCK_DURATION

    def update(self, current_pattern):
        self.history.append(current_pattern)

        if len(self.history) < 3:
            return current_pattern

        pattern_counts = {}
        for p in self.history:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        most_common = max(pattern_counts, key=pattern_counts.get)
        frequency = pattern_counts[most_common] / len(self.history)

        if self.locked_motif and self.lock_frames > 0:
            self.lock_frames -= 1
            if frequency >= 0.8 and most_common != self.locked_motif:
                self.locked_motif = most_common
                self.lock_frames = self.LOCK_DURATION
            return self.locked_motif

        if frequency >= self.stability_threshold:
            self.locked_motif = most_common
            self.lock_frames = self.LOCK_DURATION
            return most_common

        return self.locked_motif if self.locked_motif else most_common

    def reset(self):
        self.history.clear()
        self.locked_motif = ""
        self.lock_frames = 0


class BallDetector:
    """Detects purple and green balls in a frame using dual-mask color detection."""

    def __init__(self):
        self.stable = StableDetector()
        # Pre-allocate kernels once
        self._kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
        self._kernel_large = np.ones((config.MORPH_KERNEL_LARGE, config.MORPH_KERNEL_LARGE), np.uint8)

    def _clean_mask(self, mask):
        """Morphological cleanup to remove noise and fill gaps."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=config.MORPH_OPEN_ITERATIONS)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_large, iterations=config.MORPH_CLOSE_ITERATIONS)
        return mask

    def _filter_contour(self, contour):
        """Returns True if contour passes area, circularity, and solidity checks."""
        area = cv2.contourArea(contour)
        if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
            return False

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < config.MIN_CIRCULARITY:
            return False

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = area / hull_area
        if solidity < config.MIN_SOLIDITY:
            return False

        return True

    def detect(self, frame):
        """
        Detect all green and purple balls in a frame.

        Returns:
            balls: list of dicts with keys: color, x, y, w, h, center_x, center_y, area, contour
            stable_pattern: temporally-smoothed pattern string
            raw_pattern: this frame's raw pattern string
            masks: dict with 'green', 'purple', 'combined' binary masks
        """
        # Single color space conversion for HSV (shared by green + purple)
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green: dual-mask HSV + YCrCb
        mask_green_hsv = cv2.inRange(img_hsv, config.GREEN_HSV_LOWER, config.GREEN_HSV_UPPER)
        img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_green_ycrcb = cv2.inRange(img_ycrcb, config.GREEN_YCRCB_LOWER, config.GREEN_YCRCB_UPPER)
        mask_green = self._clean_mask(cv2.bitwise_and(mask_green_hsv, mask_green_ycrcb))

        # Purple: dual-mask HSV + YCrCb (reuse img_hsv and img_ycrcb)
        mask_purple_hsv = cv2.inRange(img_hsv, config.PURPLE_HSV_LOWER, config.PURPLE_HSV_UPPER)
        mask_purple_ycrcb = cv2.inRange(img_ycrcb, config.PURPLE_YCRCB_LOWER, config.PURPLE_YCRCB_UPPER)
        mask_purple = self._clean_mask(cv2.bitwise_and(mask_purple_hsv, mask_purple_ycrcb))

        # Combine and find contours
        mask_combined = cv2.bitwise_or(mask_green, mask_purple)
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        balls = []
        for contour in contours:
            if not self._filter_contour(contour):
                continue

            # Quick color classification via bounding-rect pixel counts
            x, y, w, h = cv2.boundingRect(contour)
            roi_green = mask_green[y:y+h, x:x+w]
            roi_purple = mask_purple[y:y+h, x:x+w]
            green_pixels = cv2.countNonZero(roi_green)
            purple_pixels = cv2.countNonZero(roi_purple)

            total = green_pixels + purple_pixels
            if total == 0:
                continue

            if green_pixels > purple_pixels and green_pixels / total > 0.3:
                color = "G"
            elif purple_pixels > green_pixels and purple_pixels / total > 0.3:
                color = "P"
            else:
                continue

            balls.append({
                "color": color,
                "x": x, "y": y, "w": w, "h": h,
                "center_x": x + w // 2,
                "center_y": y + h // 2,
                "area": cv2.contourArea(contour),
                "contour": contour,
            })

        # Sort left to right
        balls.sort(key=lambda b: b["center_x"])
        raw_pattern = "".join(b["color"] for b in balls)
        stable_pattern = self.stable.update(raw_pattern)

        masks = {"green": mask_green, "purple": mask_purple, "combined": mask_combined}
        return balls, stable_pattern, raw_pattern, masks

    def draw_detections(self, frame, balls, stable_pattern="", raw_pattern=""):
        """Draw detection overlays on a frame copy and return it."""
        output = frame.copy()
        green_count = 0
        purple_count = 0

        for ball in balls:
            if ball["color"] == "G":
                green_count += 1
                dc = config.COLOR_GREEN_DISPLAY
                label = f"G{green_count}"
            else:
                purple_count += 1
                dc = config.COLOR_PURPLE_DISPLAY
                label = f"P{purple_count}"

            if "contour" in ball:
                cv2.drawContours(output, [ball["contour"]], -1, dc, 2)
            else:
                x, y, w, h = ball["x"], ball["y"], ball["w"], ball["h"]
                cv2.rectangle(output, (x, y), (x + w, y + h), dc, 2)
            cv2.putText(output, label, (ball["x"], ball["y"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dc, 2)

        cv2.putText(output, f"Stable: {stable_pattern} ({len(stable_pattern)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_MATCH, 2)
        cv2.putText(output, f"Raw: {raw_pattern}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(output, f"G:{green_count} P:{purple_count}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1)

        return output
