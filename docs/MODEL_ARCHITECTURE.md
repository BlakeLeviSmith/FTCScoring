# Model Architecture: RAMP Window Detection

## Overview

The scoring system uses a **window-based YOLO detection** approach. Instead of tracking balls in flight across the entire field, we define a detection window over the RAMP area and count each new ball as it rolls through.

## Architecture

```
Camera (ESP32-S3, 5ft stand)
    |
    v
[MJPEG Stream] --> [grab_loop] --> raw frames
                                      |
                                      v
                              [YOLO Detection]
                              (within RAMP window)
                                      |
                                      v
                              [Ball Sequence Tracker]
                              (count new arrivals)
                                      |
                                      v
                              [Pattern Matching]
                              (compare vs MOTIF)
                                      |
                                      v
                              [Score Calculation]
```

## Detection Window

The RAMP is a fixed physical structure on the field. The camera is mounted on a stand at a fixed position. This means:

1. **The RAMP occupies a known region** of the camera frame
2. We define a **detection window** (ROI) around the RAMP area
3. YOLO only needs to detect balls **within this window**
4. No need to track balls in the air, on the field, or anywhere else

### Window Configuration
- Set via the web dashboard (draw ROI on camera feed)
- Stored as normalized coordinates (0-1) so it works across resolutions
- Only balls detected within the window count for scoring

## Ball Counting Logic

Instead of trying to count all 9 RAMP positions simultaneously (which requires the camera to see the entire RAMP), we track balls as they **enter the RAMP through the GOAL end**:

1. A detection zone is defined at the GOAL entrance of the RAMP
2. When a new ball enters the zone, we classify it (green/purple) and append to the sequence
3. The sequence grows as balls roll in: `G`, `GP`, `GPP`, `GPPP`, etc.
4. Each new ball is checked against the expected MOTIF pattern at that position
5. Up to 9 CLASSIFIED balls, extras are OVERFLOW

### Why This Approach
- **No simultaneous 9-ball detection needed** -- balls are counted one at a time as they arrive
- **No tracking algorithms needed** -- just detect presence/absence in the zone
- **Robust to occlusion** -- balls behind other balls don't matter since we count at entry
- **Works with limited camera angle** -- only need to see the RAMP entrance, not the full RAMP
- **Simpler model** -- YOLO just needs to detect green vs purple in a small region

## Model

- **Architecture**: YOLOv8 nano (smallest, fastest)
- **Classes**: `green_ball` (0), `purple_ball` (1)
- **Input**: Cropped RAMP window region (or full frame with window filtering)
- **Training data**: Roboflow FTC DECODE datasets + own ESP32-S3 captures
- **Target performance**: >15 FPS on ESP32-S3 feed (SVGA 800x600)

## Temporal Smoothing

The `StableDetector` provides frame-to-frame stability:
- Maintains a history of the last N detection patterns
- Locks onto the most common pattern when it reaches a threshold
- Prevents flickering between frames
- Lock duration prevents rapid switching

## Scoring Flow

```
Detected sequence: [G, P, P, G, P, P, G, P, P]
Active MOTIF:      GPP -> [G, P, P, G, P, P, G, P, P]

Position 1: G == G  -> CLASSIFIED (3pts) + PATTERN MATCH (2pts)
Position 2: P == P  -> CLASSIFIED (3pts) + PATTERN MATCH (2pts)
Position 3: P == P  -> CLASSIFIED (3pts) + PATTERN MATCH (2pts)
...
Total: 9 x 3 (classified) + 9 x 2 (pattern) = 45 pts
```

## Future Improvements

- [ ] Implement detection window ROI selection in dashboard
- [ ] Ball entry detection (gate zone) for sequential counting
- [ ] Confidence-based filtering per detection zone
- [ ] Multi-camera support (both alliance RAMPs)
- [ ] AprilTag detection for automatic MOTIF selection
