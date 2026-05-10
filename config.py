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

# Optimal camera settings for FPS (applied on startup via /control endpoint)
#
# Freenove ESP32-S3 framesize enum (esp32-camera library):
#   OV2640 (2MP):
#     6=QVGA(320x240), 10=VGA(640x480), 11=SVGA(800x600), 12=XGA(1024x768)
#   OV5640 (5MP) — same board, higher-res module:
#     12=XGA(1024x768), 13=HD(1280x720), 14=SXGA(1280x1024),
#     15=UXGA(1600x1200), 16=FHD(1920x1080)
#     Note: FHD may drop FPS to 5-8 over MJPEG. SXGA/UXGA is the sweet spot.
#
# If the camera doesn't accept a framesize, it silently falls back to its
# max — check the [CAM] log line to confirm actual resolution.
ESP32_DEFAULT_FRAMESIZE = 13    # HD 720p (1280x720) — 4x more pixels than replay footage
ESP32_DEFAULT_QUALITY = 18    # JPEG quality (8-63, lower=better but slower. 18=good balance for HD streaming)
ESP32_FLIP_IMAGE = True       # Flip image vertically (True if camera mounted upside-down)

# ============= REPLAY =============
# Replay videos get downscaled to match live ESP32 stream quality so the
# YOLO model sees the same pixel density it will at competition. HD 720p
# matches ESP32_DEFAULT_FRAMESIZE=13 above. Aspect ratio preserved; if a
# source is already at or below this height it is left untouched.
REPLAY_TARGET_HEIGHT = 720

# Simulate the lossy MJPEG compression of the live ESP32 stream by doing a
# JPEG encode/decode round-trip on every replay frame BEFORE it reaches the
# detector. Without this, replay frames are clean H.264 → cleaner than what
# the camera actually delivers, and the model can look more accurate than
# it really is. With this on, you train your eye on artifacts you'll
# actually see live.
#
# REPLAY_SIM_JPEG_QUALITY uses OpenCV's 0–100 scale (higher = better).
# ESP32 quality is on a reversed 0–63 scale (lower = better); rough mapping:
#   ESP32 q=10 ≈ OpenCV q=80   (high-quality)
#   ESP32 q=18 ≈ OpenCV q=60   (current default — matches ESP32_DEFAULT_QUALITY)
#   ESP32 q=30 ≈ OpenCV q=40   (very lossy)
# Tune until the replay artifacts look like the live stream.
REPLAY_SIMULATE_LIVE_COMPRESSION = False
REPLAY_SIM_JPEG_QUALITY = 60

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

# ============= YOLO DETECTION (MINIMAL BASELINE) =============
# This is the "vanilla" pipeline: stock YOLO confidence, stock BoT-SORT
# tracker, stock 640×640 model input, no per-class filter, no
# position-dependent filter, no green saturation boost. The point of
# the baseline is to see what the trained model alone does so we can
# add helpers back ONE AT A TIME and measure each.
#
# Knobs that ARE exposed in the live tuning UI:
#   - YOLO_CONFIDENCE
#   - YOLO_IOU_THRESHOLD
#   - ROI_CROP_INPUT_SIZE
#   - YOLO_CONF_FAR / YOLO_CONF_NEAR  (only active when FAR_REGION_FRACTION > 0)
#   - YOLO_TTA
#   - REPLAY_SIMULATE_LIVE_COMPRESSION / REPLAY_SIM_JPEG_QUALITY
#
# The rest (per-class threshold, far-region split, custom tracker yaml,
# green boost) are disabled here — flip them on individually below as
# you re-introduce each helper.
YOLO_MODEL_PATH = "training/runs/detect/ftc_motion_v1/weights/best.pt"
YOLO_CONFIDENCE = 0.25             # standard YOLO confidence (was 0.08 + secondary filters)
YOLO_IOU_THRESHOLD = 0.45

# Cap on the YOLO inference imgsz. We feed the ROI crop natively (no
# forced square resize) and ultralytics picks imgsz = next multiple of
# 32 ≥ longest side, capped here. Bigger = more HD detail preserved at
# inference; cost scales ~ imgsz².
#   640  — fastest, downscales most HD ROIs
#   960  — preserves more pixels for typical HD ROIs
#   1280 — preserves a 720p ROI fully (current default)
#   1600 — preserves slightly larger ROIs (e.g. SXGA crops)
YOLO_MAX_IMGSZ = 1280

# Per-class confidence filter (None = disabled — stock baseline).
# Set to e.g. {"G": 0.08, "P": 0.12} to accept weaker green detections.
YOLO_CONF_PER_CLASS = None

# Position-dependent confidence (FAR_REGION_FRACTION = 0 disables the
# split — stock baseline). Set to e.g. 0.45 to start applying CONF_FAR
# to the top part of the ROI crop.
YOLO_FAR_REGION_FRACTION = 0
YOLO_CONF_FAR = 0.03
YOLO_CONF_NEAR = 0.12

# Test-time augmentation: runs YOLO inference at 3 scales and merges
# results. Off in baseline. ~3× per inference call.
YOLO_TTA = False

# BoT-SORT tracker config. "botsort.yaml" is the ultralytics stock file.
# Switch to "botsort_ftc.yaml" once we want looser thresholds + longer
# track_buffer for moving-ball ReID.
YOLO_TRACKER_CONFIG = "botsort.yaml"

# Green saturation/value boost (0 = disabled — stock baseline).
GREEN_SAT_BOOST = 0
GREEN_VAL_BOOST = 0

# ============= TRIPWIRE COUNTER =============
# Velocity-projected matching: each frame, every active track's position
# is projected forward using its last known velocity, and new detections
# within TRIPWIRE_MATCH_RADIUS_PX of any projection are treated as the
# same ball continuing (no new count). Unmatched detections start a new
# track and increment the count by 1.
#   - Smaller radius → over-count (one fast ball gets split into multiple)
#   - Larger radius  → under-count (two distinct balls merged into one)
#   - Aim for ~1 ball diameter at full HD (typically 30–50 px).
TRIPWIRE_MATCH_RADIUS_PX = 40

# Frames a track survives without a detection update before being
# retired. Bridges short YOLO drops mid-pass. At 15fps:
#   3 = 0.2s (current default, generous enough for a 1-frame blink)
#   1 = strict (any drop = new track on re-detect)
TRIPWIRE_MAX_MISSES = 3

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
