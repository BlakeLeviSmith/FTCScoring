# FTCScoring — DECODE Vision Auto-Scoring

Real-time vision system that watches an FTC DECODE field through a single
overhead camera and scores each alliance's ramp automatically by counting
classified vs. overflow ball crossings.

**Where the system stands today** (see `results/*.json` for raw data):

|             | total |
|-------------|-------|
| Ground-truth events across 4 labeled match clips | 529 |
| Auto events fired                                | 384 |
| True positives (event-aligned within ±5 sec)     | 359 |
| **Precision**                                    | **93 %** |
| **Recall**                                       | **68 %** |
| **F1**                                           | **0.79** |

When the system fires, it is almost always correct. The remaining work is
in recall (cluster handling on the gate line + the overflow line, which is
currently 0 % because of a separate detector issue).

---

## Where to look — by importance for review

The repo separates source, data, tooling, and external/legacy material so
a reviewer doesn't have to dig.

### 1. Pipeline (the actual system) — **review priority**

These eight Python modules are the core. The flow is
`frame → YOLO → CSRT trackers → tripwire counter → scorer`.

| File | Role |
|---|---|
| [`app.py`](app.py) | Flask server. Owns the threads (grab loop + process loop), routes the dashboard endpoints, wires the pipeline together for both alliances, persists configs. ~3000 lines but most is glue. |
| [`yolo_detector.py`](yolo_detector.py) | YOLOv8/v11 wrapper. Loads the trained model (`training/runs/detect/ftc_motion_v2/weights/best.pt`), filters detections by per-class confidence, returns balls in the ROI's coordinate space. |
| [`csrt_tracker.py`](csrt_tracker.py) | Multi-ball CSRT tracker per alliance. Maintains track IDs across frames, handles ghost-track resurrection (a ball briefly lost mid-drop keeps its ID when re-acquired), de-duplicates near-coincident trackers. The "tracking layer." |
| [`tripwire_counter.py`](tripwire_counter.py) | Counts unique track IDs whose trail crosses a user-drawn line polygon. Two lines per alliance: classified (gate entry) + overflow. Includes per-clip `start_frame` gating for clips with unstable openings. |
| [`scorer.py`](scorer.py) | Scoring math: classified × 3 + pattern bonus × 2 + overflow × 1, MOTIF (GPP / PGP / PPG) consensus over the last few frames. |
| [`detector.py`](detector.py) | Stable-detection wrapper (temporal smoothing). Mostly legacy / lightweight; pipeline goes through `yolo_detector.py` for live mode. |
| [`ramp_tracker.py`](ramp_tracker.py) | Older "full" tracker with per-ball state machine (settling registry, eviction). Kept as a tracker backend (`--tracker full`) but not the default. |
| [`config.py`](config.py) | All tunable parameters (YOLO confidence/IoU thresholds, CSRT lifetimes, tripwire memory frames, etc.) with comments explaining each knob. **Read this first** to understand what's adjustable. |

### 2. Tooling, UI, benchmark — secondary

| File | Role |
|---|---|
| [`templates/index.html`](templates/index.html) | The single-page dashboard. Big file because it bundles HTML + CSS + JS — but it's only the operator UI. Has: live feeds, ROI/zone drawing, scoring panels, ground-truth labeling mode, benchmark comparison card. Reviewer can skim. |
| [`benchmark_all.py`](benchmark_all.py) | Headless batch runner. Walks every clip with a matching `labels/<name>.json`, spawns the server in `--auto-run --no-loop` mode, captures `results/<name>.json`, prints a summary table. |

### 3. Data the system produces / consumes — read but don't review

These are JSON inputs/outputs, not code. Reviewing them confirms the
system actually ran.

| Folder | What's in it |
|---|---|
| [`labels/`](labels/) | Ground-truth events, recorded by clicking through each clip in `--label` mode. One JSON per clip with per-click frame + timestamp + alliance + line. |
| [`results/`](results/) | Auto-counter output per clip: per-cell GT vs auto vs TP/FP/FN counts + the full lists of matched / missed / false-pos events. |
| [`roi_configs/`](roi_configs/) | Per-clip zone polygons (different camera angles need different ROIs). Optional `start_frame` field to skip the first N frames of a clip (camera-shift / pre-match). |
| [`roi_config.json`](roi_config.json) | The "global" ROI — used when no per-clip file exists yet. Acts as the seed for new clips. |

### 4. Trained model — ours, but big

| Path | What |
|---|---|
| `training/runs/detect/ftc_motion_v2/weights/best.pt` | The active YOLOv8s model (v2 of our motion-blur-augmented training). 19 MB; whitelisted in `.gitignore` so it's part of the repo. |
| [`training/`](training/) | Training scripts (`train.py`, `download_data.py`), dataset config (`dataset.yaml`), GPU-training notes (`GPU_TRAIN.md`). The raw images/labels and per-epoch checkpoints are gitignored — see [`docs/TRAINING_DATA.md`](docs/TRAINING_DATA.md) for sources. |
| `bytetrack_ftc.yaml`, `botsort_ftc.yaml` | Tuned tracker configs for ultralytics' built-in trackers. Used when `TRACKER_BACKEND="bytetrack"` in `config.py`. The default backend is `csrt` (our `csrt_tracker.py`). |

### 5. Not ours / external — skip

| Path | What |
|---|---|
| [`firmware/CameraWebServer/`](firmware/CameraWebServer/) | Stock Freenove ESP32-S3 CAM firmware. Untouched third-party code, included only so the camera setup is reproducible. |
| [`archive/`](archive/) | Older approaches we pivoted away from. `archive/hsv_pivot/` was an alternative HSV-blob counting attempt; `archive/app.py` etc. are pre-CSRT versions of the pipeline kept for context. |

---

## How to run

### Live (ESP32 camera, real match)

```bash
python app.py            # default: ESP32 WiFi at 192.168.4.1
```

### Replay a clip

```bash
python app.py --replay "match_footage/Qualification 1 of 53 - Field 1.mp4"
```

Open `http://localhost:8089`, draw the alliance ROIs and tripwire lines on
the dashboard, hit START MATCH. Live counts appear in the alliance panels.

### Collect ground truth (one-time per clip, per alliance)

```bash
python app.py --replay "match_footage/<clip>.mp4" --label --alliance red
# Q = +1 R-classified, W = +1 R-overflow, Z = undo, Space = play/pause
# Saves to labels/<clip>.json after every click.
python app.py --replay "match_footage/<clip>.mp4" --label --alliance blue
```

### Benchmark — auto-run all labeled clips end-to-end

```bash
python benchmark_all.py                # all labeled clips
python benchmark_all.py --skip-existing # only clips without results yet
```

Writes `results/<clip>.json` per clip and prints a summary table.

---

## Architecture diagram (at a glance)

```
   ESP32-S3 cam ──► grab thread ──► latest_frame
                                         │
                                         ▼
   process thread ───► yolo_detector ───► detections
                              │
                              ▼
                       csrt_tracker  ◄── ghost-track resurrection
                              │
                              ▼
                      tripwire_counter ── classified line
                              │           overflow line
                              ▼
                          scorer ─────────► dashboard
                                            (Flask + index.html)
```

`app.py` orchestrates the threads. `csrt_tracker.py` and
`tripwire_counter.py` are the two layers most worth reading carefully —
that's where recall is made or lost.

---

## Footage

The four match clips in `match_footage/` are ~110 MB each and are
gitignored. To reproduce the benchmark you'd need to drop the same MP4s
in that folder. Filenames the labels expect:

- `355_Point_Match.mp4`
- `Qualification 1 of 53 - Field 1.mp4`
- `Qualification 9 of 53 - Field 1 - 20260307 115206.mp4`
- `Qualification 13 of 53 - Field 1 - 20260307 121004.mp4`

These are San Diego FTC qualification recordings with consistent overhead
camera angles (`docs/TRAINING_DATA.md` has more on sources).

---

## Documentation

Living docs in [`docs/`](docs/):

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design, threading model
- [`MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) — detection window + ball counting
- [`SCORING_RULES.md`](docs/SCORING_RULES.md) — DECODE scoring reference
- [`TRAINING_DATA.md`](docs/TRAINING_DATA.md) — dataset sources + strategy
- [`HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) — ESP32-S3 setup
