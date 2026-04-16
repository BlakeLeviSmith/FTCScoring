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
                                │  yolo_detector.py (YOLODetector)│
                                │  └── YOLOv8n inference          │
                                │                                 │
                                │  detector.py (StableDetector)   │
                                │  └── Temporal smoothing         │
                                │                                 │
                                │  scorer.py (ScoreKeeper)        │
                                │  └── MOTIF pattern matching     │
                                └─────────────────────────────────┘
                                            │
                                    Web Dashboard (:8089)
                                    ├── Raw / YOLO Detection feeds
                                    ├── Live score display
                                    ├── YOLO confidence/IOU tuning
                                    └── MOTIF selector
```

## File Structure

```
FTCScoring/
├── app.py                  # Flask server — stream capture, YOLO detection loop, web API
├── config.py               # All tunable parameters (scoring, camera, YOLO settings)
├── detector.py             # StableDetector (temporal smoothing)
├── yolo_detector.py        # YOLODetector (YOLOv8n inference)
├── scorer.py               # ScoreKeeper — MOTIF matching, point calculation
├── requirements.txt        # Python dependencies
├── probe_stream.py         # Utility to discover ESP32 stream URLs
├── templates/
│   └── index.html          # Web dashboard (feeds, scores, YOLO tuning)
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
│   ├── MODEL_ARCHITECTURE.md # Detection window + ball counting design
│   ├── TRAINING_DATA.md    # Dataset sources and strategy
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
2. **process_loop** (Thread 2): Reads latest frame, runs YOLO detection, pre-encodes output JPEG, publishes results under `output_lock`
3. **Flask** (Main thread): Serves dashboard, streams pre-encoded JPEGs to browser, exposes JSON API

### YOLO Detection Pipeline
1. Run YOLOv8n inference on frame (configurable confidence and IOU thresholds)
2. Parse bounding boxes with class (green_ball=0, purple_ball=1) and confidence
3. Sort balls left-to-right, build pattern string
4. Run StableDetector for temporal smoothing (10-frame history, 60% consensus)
5. Returns: `(balls, stable_pattern, raw_pattern, masks)`

### Scoring
- `scorer.py` receives detected color list, compares against active MOTIF
- CLASSIFIED (on ramp, max 9): 3 points each
- OVERFLOW (beyond 9): 1 point each
- PATTERN match (color matches MOTIF at same position): 2 points each

## CLI Usage

```bash
python app.py                              # Default: ESP32 WiFi, YOLO detection
python app.py --capture                    # Use /capture polling (more compatible)
python app.py --usb                        # USB webcam (index 0)
python app.py --usb 2                      # USB webcam at index 2
python app.py --yolo-model path/to/best.pt # Custom YOLO model path
python app.py --stream-url URL             # Override stream URL
python app.py --port 9000                  # Custom web server port
```

## Web Dashboard Features

- **Video Feeds**: Raw camera feed + YOLO detection overlay feed
- **Live Scores**: CLASSIFIED count, OVERFLOW count, pattern matches, total points
- **MOTIF Selector**: GPP / PGP / PPG buttons
- **YOLO Tuning Panel** (collapsible):
  - Confidence threshold slider (live update)
  - IOU threshold slider (live update)
  - Model path display
- **Camera Controls**: Resolution, JPEG quality
- **FPS Display**: Camera input FPS and YOLO processing FPS

## Connection Monitoring

Built-in `ConnectionMonitor` prints reports every 2 minutes:
- Total frames captured
- FPS stats (avg, min, max) for the interval
- Disconnect count and duration
- Overall availability percentage

## Key Design Decisions

1. **YOLO-only detection**: Replaced color-based HSV/YCrCb detection with YOLOv8n for robustness across lighting conditions
2. **Window-based detection**: Define ROI over RAMP area, detect balls only within window (see MODEL_ARCHITECTURE.md)
3. **Pre-encoded JPEG output**: Process thread encodes once, Flask generators do zero OpenCV work
4. **Raw MJPEG parsing**: Faster than cv2.VideoCapture — parse JPEG markers directly from HTTP stream
5. **StableDetector temporal lock**: Prevents flicker — holds detected pattern for 15 frames after consensus
6. **Safe camera configuration**: Probe stream first, configure after connection confirmed, wait for stabilization
