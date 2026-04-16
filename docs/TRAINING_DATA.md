# Training Data Sources for FTC DECODE Ball Detection

## Primary Dataset

### FTC DECODE 2025: Artifacts (Roboflow)
- **URL**: https://universe.roboflow.com/robotics-m4jsb/ftc-decode-2025-artifacts-c3rys
- **Images**: ~2,391
- **Classes**: 3 -- `green`, `purple`, `negative`
- **License**: CC BY 4.0
- **Format**: YOLOv8 export available
- **Pre-trained model**: Version 3 (YOLOv11)
- **Download**:
  ```python
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_API_KEY")
  project = rf.workspace("robotics-m4jsb").project("ftc-decode-2025-artifacts-c3rys")
  version = project.version(3)
  dataset = version.download("yolov8")
  ```

## Additional Roboflow Datasets

### SolarFlare Artifacts-Decode (~2,169 images)
- **URL**: https://universe.roboflow.com/solarflare/artifacts-decode-xyqak
- Independent project (not a fork), may have different viewpoints/lighting
- Has v1 model available

### Decode Artifact Detection (~1,487 images)
- **URL**: https://universe.roboflow.com/decode-artifact-detection/ftc-decode-2025-artifacts-c3rys-aehrp
- Fork of the primary dataset (earlier version)

### Ball Detection Fork (~2,391 images)
- **URL**: https://universe.roboflow.com/ball-detection-jp4qt/ftc-decode-2025-artifacts-c3rys-i97oo
- Same image count as primary -- likely a direct fork

### FTC Fork (~1,487 images)
- **URL**: https://universe.roboflow.com/ftc-vrxfv/ftc-decode-2025-artifacts-c3rys-atuap
- Earlier version fork

### Decode by fdsgfd (different class names)
- **URL**: https://universe.roboflow.com/fdsgfd-diipt/decode-ryh3x-ojzwi
- Classes: `Green-Artifact`, `Purple-Artifact` (no negative class)
- May need class name remapping

### Artifact Detection by FTC 18840 (~36 images)
- **URL**: https://universe.roboflow.com/ftc-18840-into-the-deep/artifact-detection-0znmw/dataset/1
- Very small, early-season contribution

## Match Footage for Training/Testing

### Official FIRST Videos
- **Game Animation**: https://youtu.be/LCqWA6gSCXA -- best overview of ball flow through RAMP
- **Field Walkthrough**: https://youtu.be/lW0NzkG2zAo -- close-up of physical field hardware

### Competition Match Archives
- **The Orange Alliance**: https://theorangealliance.org -- match data with video links
- **FTCScout**: https://ftcscout.org/events/2025 -- event listings with video
- **First Preview Event (Texas)**: https://ftcscout.org/events/2025/USTXFMS1/matches

### Regional YouTube Sources (Livestream Archives)
- North Texas FTC -- automated per-match uploads via obs-ftc-stream-manager
- Hawaii FIRST Robotics: https://www.youtube.com/@hawaiifirstrobotics
- Oregon ORTOP, Minnesota High Tech Kids
- RoboZone Show (Facebook): match highlights from various qualifiers

### Finding Bulk Match Videos
The **obs-ftc-stream-manager** tool (https://github.com/FIRST-South-Carolina/obs-ftc-stream-manager) is used by many FTC event organizers to auto-upload individual match clips to YouTube and link them on The Orange Alliance. Events using this tool are the best source for per-match recordings.

### Camera Angle Notes
- Standard FTC event cameras: elevated side-angle, 8-12ft high, covering full 12x12ft field
- RAMP visibility varies by which alliance faces camera
- **Our camera setup** (ESP32-S3, 5ft stand, aimed at RAMP) will NOT match competition footage angles
- Competition footage is best for testing/validation, not primary training
- For training, capture frames with our own camera using `training/capture_frames.py`

## GitHub Resources
- **google/ftc-object-detection**: https://github.com/google/ftc-object-detection -- official Google FTC ML library (MobileNet V1 + SSD)
- **TGR-12682/FTC-Decode-2025-2026-V2**: https://github.com/TGR-12682/FTC-Decode-2025-2026-V2 -- team repo, may have vision code

## Recommended Strategy

### For Training
1. Start with the primary Roboflow dataset (2,391 images, 3 classes)
2. Supplement with SolarFlare dataset for diversity (~2,169 more images)
3. Capture additional frames from our ESP32-S3 camera for domain-specific data
4. Consider combining the "fdsgfd" dataset after remapping class names

### For Testing/Validation
1. Extract frames from competition match footage at regular intervals
2. Use `training/capture_frames.py` to capture frames from our own camera setup
3. Hand-label a test set from real match conditions for evaluation

### Class Mapping
All datasets use 2 main classes (plus optional "negative"):
| Our Classes | Roboflow Primary | fdsgfd Dataset |
|-------------|-----------------|----------------|
| `green_ball` (0) | `green` | `Green-Artifact` |
| `purple_ball` (1) | `purple` | `Purple-Artifact` |
| -- | `negative` | -- |
