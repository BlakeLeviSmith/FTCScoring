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
# Default ON because ftc_motion_v2 was trained with ImageCompression
# augmentation in the pipeline — turning sim OFF means the model sees
# *cleaner* input than it was trained on (which actually hurts accuracy
# because the model expects MJPEG artifacts in the wild).
REPLAY_SIMULATE_LIVE_COMPRESSION = True
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
YOLO_MODEL_PATH = "training/runs/detect/ftc_motion_v2/weights/best.pt"
# YOLO_CONFIDENCE is the FIRST-PASS filter applied at detection time.
# Anything below this NEVER reaches ByteTrack. Keep it LOW (~0.05) so
# ByteTrack's recovery-pass (track_low_thresh in bytetrack_ftc.yaml,
# default 0.10) actually has weak detections to work with — that's
# what saves a track during a low-confidence frame instead of letting
# it die. Higher values here defeat the whole point of ByteTrack's
# two-pass matching.
YOLO_CONFIDENCE = 0.35
# NMS IoU threshold — pairs of detections whose IoU exceeds this get the
# lower-confidence one suppressed. Higher = less aggressive suppression
# = touching balls (which overlap a lot) BOTH survive instead of one
# being killed. 0.45 is a generic default; for densely-packed objects
# like our balls 0.7 leaves real ball-pairs alone while still killing
# spurious duplicate boxes on a single ball.
# Bumped 0.50 → 0.75 to fix the cluster-undercount: when a lost ball
# re-appears at the gate touching a neighbor, NMS at 0.50 suppressed
# the second box, so the cluster matched only ONE existing tracker
# (+1 instead of +2). At 0.75 NMS only kills near-duplicate boxes on
# the same ball; touching pairs both survive and get their own tids.
YOLO_IOU_THRESHOLD = 0.75

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

# Tracker config passed to ultralytics' model.track(). Each per-stream
# tracker maintains track_id continuity frame-to-frame; the tripwire
# counter then just counts unique track_ids that crossed each zone.
#
# Options:
#   "bytetrack_ftc.yaml" (default) — ByteTrack with tuned thresholds.
#                                     Two-pass matching recovers tracks
#                                     during low-confidence frames.
#   "botsort_ftc.yaml"             — BoT-SORT with our tunes.
#   "bytetrack.yaml" / "botsort.yaml" — ultralytics stock defaults.
YOLO_TRACKER_CONFIG = "bytetrack_ftc.yaml"

# Master switch for the tracker layer.
#   "csrt"      — OpenCV CSRT correlation-filter trackers per ball.
#                 YOLO is used only for detection + anchoring; each
#                 ball follows its appearance through pixels frame-to-
#                 frame. Best for fast, unpredictable motion (bounces,
#                 direction reversals) because there's no Kalman
#                 extrapolation — the tracker just chases pixels.
#   "bytetrack" — ultralytics' model.track() with the yaml above.
#                 Tracking-by-detection: relies on YOLO firing a clean
#                 detection every frame, then associates via Kalman +
#                 Hungarian. Good when detection is consistent.
TRACKER_BACKEND = "csrt"

# CSRT tracker tuning (only used when TRACKER_BACKEND="csrt").
CSRT_MAX_LOST_FRAMES = 10          # CSRT internal-failure frames before retire
CSRT_MATCH_IOU = 0.20              # min IoU for new YOLO detection ↔ tracker
# Bumped 20 → 60 (~2s at 30fps) to fix the stagnant-ball overcount:
# a ball settles in the zone, CSRT keeps tracking it, but YOLO may
# briefly stop firing on it (NMS suppression by neighbors, mask
# flicker). At 20 the tracker retired, then a fresh tid spawned on
# the same physical ball later → tripwire counted it again. At 60 the
# tracker survives those gaps and keeps its original tid, so the
# tripwire stays at +1.
CSRT_MAX_FRAMES_WITHOUT_YOLO = 60
# Second-pass center-distance threshold (px) for matching an unmatched
# YOLO detection to an existing tracker. IoU alone fails when the
# detection drifts off-bbox; a center within this radius is treated as
# the same ball and reabsorbed into the existing tracker rather than
# spawning a duplicate. ~1 ball diameter at typical playback zoom.
CSRT_MATCH_CENTER_PX = 35

# Same-color trackers within this many pixels of each other are merged
# (younger one retired silently). Fixes the duplicate-tracker overcount
# you saw in your debug trace where 4 trackers covered 2 actual balls
# at (90,185). Lower = more aggressive merge; raise if balls genuinely
# touch shoulder-to-shoulder and you want each kept distinct.
# A single ball is ~2*sqrt(area/pi) ≈ 42px wide, so 25px = ~half a ball.
CSRT_DEDUP_RADIUS_PX = 25

# Ghost-track resurrection — when CSRT retires a tracker, push its last
# (position, color, tid) to a ghost list. If a new YOLO detection
# arrives within CSRT_GHOST_MATCH_RADIUS_PX of a ghost (and color
# matches if required) within CSRT_GHOST_MAX_FRAMES, we RESURRECT the
# tracker under the ORIGINAL track_id instead of spawning a new one.
# This is the simple-and-direct fix for "ball settles, CSRT loses it,
# re-acquires as new tid" — the tid stays the same so the tripwire
# counter doesn't double-count.
# Bumped 45 → 90 (~3s) to back up the patient YOLO-confirmation guard:
# even if a tracker DOES retire, the ghost lives long enough that the
# eventual re-detection still resurrects under the original tid
# instead of spawning a fresh one (which the tripwire would re-count).
CSRT_GHOST_MAX_FRAMES = 90
# Wider search radius — the ball drops vertically through the gate
# zone quickly (~80-100 px in 3-5 frames at 30fps) and reappears on
# the ramp below. The velocity projection in _find_matching_ghost
# handles most of the displacement; this radius is the slop budget
# around the projected point. 90 px covers ~2 ball diameters.
CSRT_GHOST_MATCH_RADIUS_PX = 90
CSRT_GHOST_REQUIRE_COLOR = True

# Hard cap on concurrently-active CSRT trackers per alliance. Each tracker
# runs CSRT.update() on every frame at ~20-50ms; without a ceiling, a
# burst of false-positive detections spawns dozens of trackers and
# processing FPS collapses (observed 18 -> 4 fps with ~45% drop rate).
# 12 covers the realistic max of 9 RAMP balls + a few in flight, leaves
# headroom for one over-detection per ball, and refuses to spawn beyond
# that. Real balls will still be matched via the ghost-resurrection path.
CSRT_MAX_ACTIVE_TRACKS = 12

# Green saturation/value boost (0 = disabled — stock baseline).
GREEN_SAT_BOOST = 0
GREEN_VAL_BOOST = 0

# ============= TRIPWIRE COUNTER =============
# Track-ID-based counting. Each frame, ultralytics' tracker
# (ByteTrack/BoT-SORT, see YOLO_TRACKER_CONFIG above) assigns a
# persistent track_id to every detection. Tripwires just record the
# set of unique track_ids ever seen inside their polygon, and the
# count is len(seen_ids).
#
# All inter-frame association is delegated to the ByteTrack/BoT-SORT
# Kalman filter + Hungarian matching + low-confidence-recovery — no
# velocity gates, bounce radii, etc. needed at this layer.
#
# How long to remember a seen track_id (frames). After this many
# frames since last seen, the id is forgotten so a re-issued id (e.g.
# after the source ball physically left and ByteTrack recycled the
# number) can register as a fresh count. At 30fps, 600 frames = 20s,
# longer than any plausible "ball that's still in flight."
TRIPWIRE_TRACK_MEMORY_FRAMES = 600

# Trail visualization (debug overlay): how many recent positions to
# remember per active track for drawing the polyline trail. ~30 frames
# at 30fps = 1s of motion history per track.
TRIPWIRE_TRAIL_LENGTH = 30

# Track maturity requirement: a track must have been visible (anywhere
# in the ROI) for at least this many frames before its appearance in a
# tripwire counts. Bridges the case where ByteTrack assigns a tentative
# track_id on the very first detection of a ball — that id may be
# revised in the next 1-2 frames as the tracker promotes it from
# "tentative" to "tracked." Counting only after N frames lets the id
# stabilize first.
#
# At 30fps:
#   1 = no maturity requirement (current ByteTrack behavior)
#   3 = wait 100ms (~6 frames at 60fps replay) — light requirement
#   5 = wait 167ms — recommended for balls that come straight out of
#       the goal and are immediately at the tripwire
# Lowered from 5 → 2: CSRT + ghost-resurrection keeps the same track_id
# across detection gaps, so there's no tentative-id flicker to wait out
# the way ByteTrack required. 5 frames was killing fast normal passes
# (ball in ROI for 6-8 frames total, barely matures before exiting).
TRIPWIRE_MIN_TRACK_AGE_FRAMES = 2

# Settled-Ball Registry (anti-double-count for stationary balls).
#
# A ball that has been counted by a tripwire and then sits motionless
# (a "settled" ball) is recorded with its position + color. If CSRT
# later loses it (collision, occlusion, etc.) and a NEW track_id is
# spawned at the same spot, the registry deduplicates — the new track
# is recognized as the same physical ball and isn't counted again.
#
# A track is considered SETTLED when its last
# TRIPWIRE_SETTLE_FRAMES_REQUIRED positions have all moved less than
# TRIPWIRE_SETTLE_MOTION_THRESHOLD_PX pixels.
#
# A new track_id is matched against the registry within
# TRIPWIRE_SETTLED_MATCH_RADIUS_PX of any registered position; same
# color required if TRIPWIRE_SETTLED_REQUIRE_COLOR=True.
#
# Registry entries that haven't been re-confirmed in
# TRIPWIRE_SETTLED_EVICTION_FRAMES are dropped (handles balls that
# physically leave the zone).
TRIPWIRE_SETTLE_MOTION_THRESHOLD_PX = 8
TRIPWIRE_SETTLE_FRAMES_REQUIRED = 5
TRIPWIRE_SETTLED_MATCH_RADIUS_PX = 30
# TRANSIT entries (ball moving through, never settled) evict fast so
# they don't block subsequent balls passing through the same area.
# At 30fps, 10 frames ≈ 0.33s — generous enough to bridge a 1-2 frame
# YOLO blink, short enough that the next ball through doesn't dedupe.
TRIPWIRE_TRANSIT_EVICTION_FRAMES = 10
# SETTLED entries (ball stopped in zone) get the long window — long
# enough to bridge multi-second CSRT losses during collisions /
# occlusion / appearance changes.
TRIPWIRE_SETTLED_EVICTION_FRAMES = 150
TRIPWIRE_SETTLED_REQUIRE_COLOR = True

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
