# FTC Scoring Vision System - Architecture

## Overview

Automated scoring system for 2025-2026 FTC DECODE. Uses a Freenove ESP32-S3 CAM mounted on a stand ~5ft high to detect purple/green ball positions on alliance RAMPs and calculate match scores in real-time via a web dashboard.

## System Diagram

```
ESP32-S3 CAM (Wi-Fi AP)         Python Server (Laptop)
┌──────────────────┐            ┌─────────────────────────────────┐
│ OV2640 Sensor    │            │  app.py (Flask)                 │
│ SVGA 800x600     │──MJPEG────>│  ├── grab_loop (stream parser)  │
│ 192.168.4.1:81   │  over WiFi │  ├── process_loop (detection)   │
│                  │            │  ├── ConnectionMonitor           │
└──────────────────┘            │  └── Web Dashboard (:8089)      │
                                │                                 │
                                │  detector.py (BallDetector)     │
                                │  ├── HSV + YCrCb dual-mask      │
                                │  ├── Morphological cleanup      │
                                │  ├── Contour filtering          │
                                │  └── StableDetector (temporal)  │
                                │                                 │
                                │  yolo_detector.py (YOLODetector)│
                                │  └── YOLOv8n inference          │
                                │     (same interface, optional)  │
                                │                                 │
                                │  scorer.py (ScoreKeeper)        │
                                │  └── MOTIF pattern matching     │
                                └─────────────────────────────────┘
                                            │
                                    Web Dashboard (:8089)
                                    ├── Raw / Processed / Mask feeds
                                    ├── Live score display
                                    ├── Detection tuning panel
                                    └── MOTIF selector
```

## File Structure

```
FTCScoring/
├── app.py                  # Flask server — stream capture, detection loop, web API
├── config.py               # All tunable parameters (thresholds, scoring, camera)
├── detector.py             # BallDetector + StableDetector (color-based detection)
├── yolo_detector.py        # YOLODetector (YOLOv8n, same interface as BallDetector)
├── scorer.py               # ScoreKeeper — MOTIF matching, point calculation
├── requirements.txt        # Python dependencies
├── probe_stream.py         # Utility to discover ESP32 stream URLs
├── templates/
│   └── index.html          # Web dashboard (feeds, scores, tuning panel)
├── training/
│   ├── dataset.yaml        # YOLO dataset config (2 classes)
│   ├── train.py            # YOLOv8n training script
│   ├── capture_frames.py   # Frame capture helper for training data
│   ├── README.md           # Training workflow guide
│   ├── images/train/       # Training images
│   ├── images/val/         # Validation images
│   ├── labels/train/       # YOLO format labels
│   └── labels/val/         # YOLO format labels
├── docs/
│   ├── ARCHITECTURE.md     # This file
│   ├── SCORING_RULES.md    # DECODE scoring reference
│   └── HARDWARE_SETUP.md   # ESP32-S3 setup guide
├── vision.py               # Legacy (robot-mounted camera)
├── vision_debug.py         # Legacy (debug tools)
├── vision_fixed.py         # Legacy (cleaner rewrite)
└── CLAUDE.md               # Agent instructions
```

## Processing Pipeline

### Two-Thread Architecture
1. **grab_loop** (Thread 1): Connects to ESP32 MJPEG stream on port 81, parses raw JPEG frames from the byte stream, stores latest frame under `frame_lock`
2. **process_loop** (Thread 2): Reads latest frame, runs detection pipeline, pre-encodes output JPEGs, publishes results under `output_lock`
3. **Flask** (Main thread): Serves dashboard, streams pre-encoded JPEGs to browser, exposes JSON API

### Detection Pipeline (Color-Based)
1. Convert frame to HSV and YCrCb color spaces (one conversion each, shared)
2. For each color (green, purple):
   - Create HSV mask via `cv2.inRange`
   - Create YCrCb mask via `cv2.inRange`
   - AND the two masks (dual-mask reduces false positives)
   - Morphological cleanup: OPEN (5x5, 1 iter) then CLOSE (7x7, 2 iter)
3. Combine masks, find contours
4. Filter contours by area (50-5000), circularity (>0.38), solidity (>0.76)
5. Classify each contour as G or P by pixel count ratio in each mask
6. Sort balls left-to-right, build pattern string
7. Run StableDetector for temporal smoothing (10-frame history, 60% consensus)

### Detection Pipeline (YOLO, Optional)
1. Run YOLOv8n inference on frame
2. Parse bounding boxes with class (green_ball=0, purple_ball=1) and confidence
3. Same sorting, pattern building, and temporal smoothing as color-based
4. Returns same interface: `(balls, stable_pattern, raw_pattern, masks)`

### Scoring
- `scorer.py` receives detected color list, compares against active MOTIF
- CLASSIFIED (on ramp, max 9): 3 points each
- OVERFLOW (beyond 9): 1 point each
- PATTERN match (color matches MOTIF at same position): 2 points each

## CLI Usage

```bash
python app.py                          # Default: ESP32 WiFi, color detection
python app.py --capture                # Use /capture polling (more compatible)
python app.py --usb                    # USB webcam (index 0)
python app.py --usb 2                  # USB webcam at index 2
python app.py --yolo                   # YOLOv8n detection (default model path)
python app.py --yolo --yolo-model best.pt  # YOLOv8n with custom model
python app.py --stream-url URL         # Override stream URL
python app.py --port 9000              # Custom web server port
```

## Web Dashboard Features

- **Video Feeds**: Raw camera, processed (with detection overlays), green mask, purple mask
- **Live Scores**: CLASSIFIED count, OVERFLOW count, pattern matches, total points
- **MOTIF Selector**: GPP / PGP / PPG buttons
- **Detection Tuning Panel** (collapsible):
  - Pixel Inspector — click raw feed to see HSV/YCrCb/RGB values
  - Green HSV + YCrCb sliders (12 sliders, live update with 100ms debounce)
  - Purple HSV + YCrCb sliders (12 sliders)
  - Contour filtering: min/max area, min circularity, min solidity
  - Auto-tune: draw ROI around balls, automatic threshold optimization
  - Save to config.py / Reset Defaults buttons
- **FPS Display**: Grab FPS and process FPS

## Connection Monitoring

Built-in `ConnectionMonitor` prints reports every 2 minutes:
- Total frames captured
- FPS stats (avg, min, max) for the interval
- Disconnect count and duration
- Overall availability percentage

## Key Design Decisions

1. **Dual-mask detection**: HSV AND YCrCb reduces false positives vs either alone
2. **Pre-encoded JPEG output**: Process thread encodes once, Flask generators do zero OpenCV work
3. **Raw MJPEG parsing**: Faster than cv2.VideoCapture — parse JPEG markers directly from HTTP stream
4. **StableDetector temporal lock**: Prevents flicker — holds detected pattern for 15 frames after consensus
5. **Detector interface pattern**: Both BallDetector and YOLODetector implement identical `detect()` / `draw_detections()` — scorer.py and app.py are detector-agnostic
6. **Safe camera configuration**: Probe stream first, configure after connection confirmed, wait for stabilization
