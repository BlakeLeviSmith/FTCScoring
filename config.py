"""
Configuration for FTC DECODE Vision Scoring System.
All tunable parameters in one place.
"""

import numpy as np

# ============= ESP32-CAM CONNECTION =============
# Connect to ESP32's WiFi first, then this stream is available
ESP32_BASE_URL = "http://192.168.4.1"
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"
ESP32_CAPTURE_URL = "http://192.168.4.1/capture"
ESP32_CONTROL_URL = "http://192.168.4.1/control"
ESP32_STATUS_URL = "http://192.168.4.1/status"

# Optimal camera settings for FPS (applied on startup)
# Freenove ESP32-S3 framesize enum (newer esp32-camera library):
#   6=QVGA(320x240), 10=VGA(640x480), 11=SVGA(800x600), 12=XGA(1024x768)
# Old ESP32-CAM framesize enum:
#   4=QVGA, 6=VGA, 7=SVGA, 8=XGA
ESP32_DEFAULT_FRAMESIZE = 11  # SVGA (800x600) for Freenove ESP32-S3
ESP32_DEFAULT_QUALITY = 15    # JPEG quality (8-63, lower=better quality but slower)

# ============= GREEN DETECTION (dual-mask HSV + YCrCb) =============
GREEN_HSV_LOWER = np.array([35, 40, 40])
GREEN_HSV_UPPER = np.array([85, 255, 255])
GREEN_YCRCB_LOWER = np.array([40, 0, 0])
GREEN_YCRCB_UPPER = np.array([220, 115, 135])

# ============= PURPLE DETECTION (HSV + YCrCb) =============
PURPLE_HSV_LOWER = np.array([134, 5, 50])
PURPLE_HSV_UPPER = np.array([179, 210, 188])
PURPLE_YCRCB_LOWER = np.array([40, 101, 100])
PURPLE_YCRCB_UPPER = np.array([135, 200, 199])

# ============= MORPHOLOGY CLEANUP =============
MORPH_KERNEL_SIZE = 5
MORPH_KERNEL_LARGE = 7
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

# ============= DETECTION PARAMETERS =============
MIN_CONTOUR_AREA = 50
MAX_CONTOUR_AREA = 5000
MIN_CIRCULARITY = 0.38
MIN_SOLIDITY = 0.76

# ============= TEMPORAL SMOOTHING =============
STABLE_HISTORY_SIZE = 10
STABLE_THRESHOLD = 0.6
STABLE_LOCK_DURATION = 15

# ============= MOTIFS =============
# Each MOTIF is a 3-color pattern repeated 3x for 9 RAMP positions
# Position 1 = Gate end, Position 9 = Square end
MOTIFS = {
    "GPP": ["G", "P", "P", "G", "P", "P", "G", "P", "P"],
    "PGP": ["P", "G", "P", "P", "G", "P", "P", "G", "P"],
    "PPG": ["P", "P", "G", "P", "P", "G", "P", "P", "G"],
}

# AprilTag IDs corresponding to each MOTIF
MOTIF_APRILTAG_IDS = {
    21: "GPP",
    22: "PGP",
    23: "PPG",
}

# ============= SCORING POINT VALUES =============
POINTS_CLASSIFIED_AUTO = 3
POINTS_CLASSIFIED_TELEOP = 3
POINTS_OVERFLOW_AUTO = 1
POINTS_OVERFLOW_TELEOP = 1
POINTS_PATTERN_MATCH = 2
POINTS_DEPOT = 1

# ============= YOLO DETECTION =============
YOLO_MODEL_PATH = "training/runs/detect/ftc_balls/weights/best.pt"
YOLO_CONFIDENCE = 0.4
YOLO_IOU_THRESHOLD = 0.45

# ============= DISPLAY =============
COLOR_GREEN_DISPLAY = (0, 255, 0)
COLOR_PURPLE_DISPLAY = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_MATCH = (0, 255, 255)
COLOR_NO_MATCH = (0, 0, 255)
