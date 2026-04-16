# FTCScoring - DECODE Vision Scoring System

## Project Summary
Automated scoring system for FTC 2025-2026 DECODE game. Uses a Freenove ESP32-S3 CAM (OV2640) on a stand ~5ft high to detect purple/green ball positions on alliance RAMPs and calculate match scores in real-time via a web dashboard.

## Key Context
- Camera: Freenove ESP32-S3 CAM, Wi-Fi AP at `http://192.168.4.1/`
- MJPEG stream: `http://192.168.4.1:81/stream` (port 81, one client at a time)
- Resolution: SVGA 800x600 (framesize=11 in Freenove S3 firmware)
- Game: DECODE — robots score purple (P) and green (G) ball ARTIFACTS into GOALs
- ARTIFACTS flow through CLASSIFIER: GOAL -> SQUARE -> RAMP -> GATE
- RAMP holds up to 9 CLASSIFIED balls in order
- 3 possible MOTIFS: GPP, PGP, PPG (each repeated 3x for 9 positions)
- CLASSIFIED = 3pts, OVERFLOW = 1pt, PATTERN match = 2pts per position

## Task Management — MANDATORY

**Always use `bd` (beads) for ALL task tracking. Never use TodoWrite, TaskCreate, or markdown files for tasks.**

### Session Start
- Run `bd ready` to see available work
- Run `bd list --status=open` to see all open issues
- Claim work with `bd update <id> --status=in_progress` before starting

### During Work
- Create a bead BEFORE writing code: `bd create --title="..." --description="..." --type=task --priority=2`
- Mark in_progress when starting: `bd update <id> --status=in_progress`
- Use `bd dep add <issue> <depends-on>` for dependencies between tasks
- Priority: 0=critical, 1=high, 2=medium, 3=low, 4=backlog

### Completing Work
- Close when done: `bd close <id>` (or `bd close <id> --reason="explanation"`)
- Close multiple at once: `bd close <id1> <id2> ...`
- Create follow-up beads for any new work discovered during implementation

### Session End
- Run `bd sync` to save state
- Ensure all completed work has closed beads
- Ensure any unfinished work has open beads with clear descriptions

### Rules
- Every non-trivial task gets a bead — no exceptions
- Never leave work unclaimed: if you start it, mark it in_progress
- Never leave work untracked: if you finish it, close the bead
- Descriptions should be detailed enough for another agent to pick up the work
- Do NOT use `bd edit` — it opens $EDITOR which blocks agents

## Architecture
- `app.py` — Flask server, two-thread architecture (grab + process), web API, connection monitor
- `config.py` — all tunable parameters (scoring, camera, YOLO settings)
- `detector.py` — StableDetector (temporal smoothing)
- `yolo_detector.py` — YOLODetector (YOLOv8n, the only detection method)
- `scorer.py` — ScoreKeeper, MOTIF pattern matching
- `templates/index.html` — web dashboard with live feeds, scores, YOLO tuning panel
- `training/` — YOLOv8n training pipeline (train.py, capture_frames.py, dataset.yaml)

See `docs/ARCHITECTURE.md` for full system design.
See `docs/MODEL_ARCHITECTURE.md` for detection window + ball counting design.
See `docs/TRAINING_DATA.md` for dataset sources and strategy.
See `docs/SCORING_RULES.md` for DECODE scoring reference.
See `docs/HARDWARE_SETUP.md` for ESP32-S3 setup.

## Code Conventions
- Python 3.10+
- OpenCV (cv2) for vision processing
- NumPy for array operations
- YOLO-only detection via ultralytics YOLOv8n
- Config values in `config.py` (not hardcoded)
- Detector interface: `detect(frame)` -> `(balls, stable_pattern, raw_pattern, masks)`

## CLI Flags
```
python app.py                    # Default: ESP32 WiFi + YOLO detection
python app.py --capture          # Use /capture polling instead of MJPEG stream
python app.py --usb [INDEX]      # USB webcam
python app.py --yolo-model PATH  # Custom YOLO model path
python app.py --stream-url URL   # Override stream URL
python app.py --port PORT        # Custom web server port
```

## Important Hardware Notes
- Freenove ESP32-S3 framesize enum: SVGA=11, VGA=10 (different from old ESP32-CAM where SVGA=7)
- Only ONE MJPEG stream client at a time — close browser tabs showing 192.168.4.1
- Power cycle ESP32 if camera gets into bad state (HTTP 500, stream won't connect)
- Camera config commands sent AFTER stream connects, with 2s stabilization delay

## Legacy Files
- `vision.py`, `vision_debug.py`, `vision_fixed.py` — old scripts for robot-mounted camera
- `probe_stream.py` — utility to discover ESP32 stream URLs
