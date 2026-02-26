# YOLO Training for FTC DECODE Ball Detection

Train a YOLOv8 nano model to detect green and purple balls on the RAMP.

## Quick Start

### 1. Install Dependencies

```bash
# CPU-only (recommended — avoids 2GB CUDA download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

For Apple Silicon GPU acceleration:
```bash
pip install torch torchvision
pip install ultralytics
# Then train with: --device mps
```

### 2. Capture Training Images

```bash
# From ESP32-CAM (connect to its WiFi first)
python training/capture_frames.py --count 200

# From USB webcam
python training/capture_frames.py --count 200 --usb
```

Capture frames with balls in various positions on the RAMP. Vary:
- Number of balls (1-9)
- Mix of green/purple
- Lighting conditions (if possible)
- Ball positions along the RAMP

### 3. Annotate Images

Use [labelImg](https://github.com/HumanSignal/labelImg) (free, local) or [Roboflow](https://roboflow.com) (web-based):

**Classes:**
| ID | Name |
|----|------|
| 0 | green_ball |
| 1 | purple_ball |

**labelImg setup:**
```bash
pip install labelImg
labelImg training/images/train/ training/labels/train/classes.txt
```

Create `training/labels/train/classes.txt`:
```
green_ball
purple_ball
```

- Draw bounding boxes around each ball
- Save in YOLO format (one `.txt` per image)

### 4. Split Train/Val

Move ~20% of images and their matching label files to the validation set:

```bash
# Example: move every 5th image to val
cd training
for f in images/train/frame_00{00,05,10,15,20,25,30,35,40}*.jpg; do
  base=$(basename "$f" .jpg)
  mv "images/train/${base}.jpg" images/val/
  mv "labels/train/${base}.txt" labels/val/ 2>/dev/null
done
```

### 5. Train

```bash
# CPU (default, ~30-60 min for 100 epochs with 200 images)
python training/train.py

# Apple Silicon GPU (faster)
python training/train.py --device mps

# More epochs
python training/train.py --epochs 200

# Resume interrupted training
python training/train.py --resume
```

Output: `training/runs/detect/ftc_balls/weights/best.pt`

### 6. Run

```bash
python app.py --yolo --yolo-model training/runs/detect/ftc_balls/weights/best.pt
```

## Directory Structure

```
training/
  dataset.yaml           # YOLO dataset config
  train.py               # Training script
  capture_frames.py      # Frame capture helper
  README.md              # This file
  images/
    train/               # Training images (.jpg)
    val/                 # Validation images (.jpg)
  labels/
    train/               # Training labels (.txt, YOLO format)
    val/                 # Validation labels (.txt)
  runs/                  # Training outputs (auto-created)
    detect/
      ftc_balls/
        weights/
          best.pt        # Best model
          last.pt        # Latest checkpoint
```

## Label Format

Each `.txt` file has one line per ball:
```
class_id  x_center  y_center  width  height
```
All values normalized 0-1 (relative to image width/height).

Example (`frame_0042.txt`):
```
0 0.45 0.62 0.035 0.045
1 0.52 0.61 0.032 0.042
1 0.60 0.63 0.030 0.040
```

## Tips

- **More data = better results.** 200+ annotated images recommended.
- **Include negative examples** — frames with no balls help reduce false positives.
- **Keep consistent camera angle** — train with the same setup you'll use in competition.
- **Small balls (14px at VGA)** — draw tight bounding boxes, don't include too much background.
- If detection is poor, try `--epochs 200` or capture more training images.
