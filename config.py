"""
Configuration for FTC DECODE Vision Scoring System.
All tunable parameters in one place.
"""

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
YOLO_MODEL_PATH = "training/runs/detect/ftc_cpu_final/weights/best.pt"
YOLO_CONFIDENCE = 0.12
YOLO_IOU_THRESHOLD = 0.45

# ============= TEMPORAL SMOOTHING =============
STABLE_HISTORY_SIZE = 10
STABLE_THRESHOLD = 0.6
STABLE_LOCK_DURATION = 15

# ============= MATCH TIMING =============
AUTO_DURATION_S = 30
TELEOP_DURATION_S = 120

# ============= RAMP TRACKER =============
GATE_COOLDOWN_FRAMES = 10  # Frames to ignore gate zone after counting a ball
RAMP_MAX_BALLS = 9         # Maximum CLASSIFIED balls on the RAMP

# When a tracked ball on the RAMP stops being detected, we need to decide
# whether it actually LEFT or whether the detector just dropped it for a bit.
# Purple balls are usually solid; green balls go missing for ~1.5–2s at a time
# in our YOLO model, so they get a longer grace window before we assume exit.
EXIT_GRACE_SEC_PURPLE = 1.0
EXIT_GRACE_SEC_GREEN = 2.5

# After grace expires, a ball is declared EXITED only if its last known
# position was within this normalized margin of the exit zone (or inside it).
# Margin is expressed as a fraction of mean(frame_w, frame_h). If the last
# position is deep inside the ramp, we assume a detector drop and keep the
# ball on the ramp (its track will re-associate when YOLO picks it up again).
EXIT_PROXIMITY_MARGIN = 0.06

# Hard ceiling: if a track has been missing for this many × its grace window,
# declare it exited even if it wasn't near the exit zone. Prevents phantom
# occupancy from permanently-lost tracks. Stable-locked balls ignore this.
EXIT_HARD_TIMEOUT_MULT = 4.0

# Rebind: YOLO/ByteTrack sometimes drops a track and re-acquires the same
# physical ball under a NEW track_id. Without rebinding, the new ID looks
# like a brand-new ball entering through the gate and gets double-counted
# (this is the "overflow keeps ticking up while the ramp sits still" bug).
# When a brand-new track_id appears, we first check whether it's close to
# any recently-lost on-ramp ball — if so, we adopt the old entry under the
# new ID instead of counting it.
REBIND_RADIUS = 0.08          # normalized (fraction of mean(w,h))
REBIND_WINDOW_SEC = 60.0      # long enough to cover an entire match period

# Stability lock: once a ball has been sitting within a small radius for
# long enough, we "lock it in" as present. Locked balls are never declared
# exited by the missing-track grace logic — they only leave if the user
# resets the match or we observe them clearly leaving the ROI. This stops
# score fluctuation when robots briefly occlude the ramp.
STABILITY_WINDOW_SEC = 1.5    # lookback window for motion variance
STABILITY_RADIUS = 0.015      # max positional jitter (normalized) to be "still"
STABILITY_UNLOCK_MOVE = 0.05  # if a locked ball reappears this far away, unlock

# ============= DISPLAY =============
COLOR_GREEN_DISPLAY = (0, 255, 0)
COLOR_PURPLE_DISPLAY = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_MATCH = (0, 255, 255)
COLOR_NO_MATCH = (0, 0, 255)
