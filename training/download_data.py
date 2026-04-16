"""
Download FTC DECODE training datasets from Roboflow and prepare for YOLOv8 training.

Usage:
    python training/download_data.py --api-key YOUR_KEY          # Download primary dataset
    python training/download_data.py --source all --api-key KEY  # Download all datasets
    python training/download_data.py --video match.mp4 --fps 2   # Extract frames from video
    python training/download_data.py --stats                     # Print current dataset stats

Datasets:
    primary    - robotics-m4jsb/ftc-decode-2025-artifacts-c3rys v3 (2,391 images, CC BY 4.0)
    solarflare - solarflare/artifacts-decode-xyqak v1

The downloaded data is remapped to match our 2-class training pipeline:
    0: green_ball
    1: purple_ball
    (Roboflow "negative" class labels are dropped)
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset registry: (workspace, project, version)
DATASETS = {
    "primary": ("robotics-m4jsb", "ftc-decode-2025-artifacts-c3rys", 3),
    "solarflare": ("solarflare", "artifacts-decode-xyqak", 1),
}

# Roboflow class names -> our class IDs
# Our pipeline: green_ball=0, purple_ball=1
# Roboflow primary: green=0, purple=1, negative=2
ROBOFLOW_CLASS_MAP = {
    "green": 0,       # -> green_ball (0)
    "purple": 1,      # -> purple_ball (1)
    "negative": None,  # dropped
}

OUR_CLASSES = {0: "green_ball", 1: "purple_ball"}


def get_api_key(args):
    """Get Roboflow API key from args or environment."""
    key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Error: Roboflow API key required.")
        print("  Provide via --api-key YOUR_KEY or set ROBOFLOW_API_KEY env var.")
        print("  Get a free key at https://app.roboflow.com/settings/api")
        sys.exit(1)
    return key


def download_dataset(workspace, project, version, api_key, dest_dir):
    """Download a dataset from Roboflow in YOLOv8 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Error: roboflow package not installed.")
        print("  pip install roboflow")
        sys.exit(1)

    print(f"Connecting to Roboflow ({workspace}/{project} v{version})...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version)

    print(f"Downloading dataset in YOLOv8 format to {dest_dir}...")
    dataset = ds.download("yolov8", location=dest_dir, overwrite=True)
    print(f"Download complete: {dataset.location}")
    return dataset.location


def remap_label_file(label_path, class_map, drop_unmapped=True):
    """Remap class IDs in a YOLO label file.

    class_map: dict mapping old_id (int) -> new_id (int or None).
               None means drop that class.
    Returns (kept, dropped) line counts.
    """
    if not os.path.isfile(label_path):
        return 0, 0

    kept, dropped = 0, 0
    new_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = class_map.get(old_id)
            if new_id is None:
                dropped += 1
                continue
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts))
            kept += 1

    with open(label_path, "w") as f:
        f.write("\n".join(new_lines))
        if new_lines:
            f.write("\n")

    return kept, dropped


def detect_class_mapping(download_dir):
    """Detect class names from downloaded data.yaml and build ID mapping.

    Returns a dict mapping old_class_id -> new_class_id (or None to drop).
    """
    import yaml

    data_yaml = os.path.join(download_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        # Roboflow sometimes nests it
        for root, dirs, files in os.walk(download_dir):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                break

    if not os.path.exists(data_yaml):
        print("Warning: No data.yaml found in download. Using default class mapping.")
        return {0: 0, 1: 1}  # Assume green=0, purple=1

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    print(f"  Roboflow classes: {names}")

    mapping = {}
    for old_id, name in names.items():
        old_id = int(old_id)
        name_lower = name.lower().strip()
        if name_lower in ROBOFLOW_CLASS_MAP:
            mapping[old_id] = ROBOFLOW_CLASS_MAP[name_lower]
        elif "green" in name_lower:
            mapping[old_id] = 0
        elif "purple" in name_lower:
            mapping[old_id] = 1
        else:
            # Unknown class (e.g. "negative") -> drop
            mapping[old_id] = None
            print(f"  Dropping class {old_id}: '{name}' (not green/purple)")

    return mapping


def find_split_dirs(download_dir):
    """Find train/valid/test image and label directories in the download.

    Roboflow YOLOv8 format typically uses:
      <root>/train/images/, <root>/train/labels/
      <root>/valid/images/, <root>/valid/labels/
      <root>/test/images/,  <root>/test/labels/  (optional)
    """
    splits = {}
    for split_name in ["train", "valid", "val", "test"]:
        img_dir = os.path.join(download_dir, split_name, "images")
        lbl_dir = os.path.join(download_dir, split_name, "labels")
        if os.path.isdir(img_dir):
            splits[split_name] = {"images": img_dir, "labels": lbl_dir}

    if not splits:
        # Try flat structure: images/ and labels/ at root with no split subdirs
        img_dir = os.path.join(download_dir, "images")
        lbl_dir = os.path.join(download_dir, "labels")
        if os.path.isdir(img_dir):
            splits["train"] = {"images": img_dir, "labels": lbl_dir}

    return splits


def copy_files_to_pipeline(splits, class_mapping, prefix=""):
    """Copy images and labels from downloaded splits into our pipeline directories.

    Maps Roboflow splits: train -> train, valid/val -> val, test -> val (merged).
    Remaps class IDs in label files. Prefixes filenames to avoid collisions.
    """
    # Map Roboflow split names to our split names
    split_map = {
        "train": "train",
        "valid": "val",
        "val": "val",
        "test": "val",  # merge test into val
    }

    stats = {"train": {"images": 0, "labels_kept": 0, "labels_dropped": 0},
             "val": {"images": 0, "labels_kept": 0, "labels_dropped": 0}}

    for rf_split, dirs in splits.items():
        our_split = split_map.get(rf_split, "train")
        dest_img = os.path.join(TRAINING_DIR, "images", our_split)
        dest_lbl = os.path.join(TRAINING_DIR, "labels", our_split)
        os.makedirs(dest_img, exist_ok=True)
        os.makedirs(dest_lbl, exist_ok=True)

        img_dir = dirs["images"]
        lbl_dir = dirs.get("labels", "")

        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(glob.glob(os.path.join(img_dir, ext)))
            image_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))

        for img_path in image_files:
            basename = os.path.basename(img_path)
            stem, ext = os.path.splitext(basename)
            dest_name = f"{prefix}{stem}" if prefix else stem

            # Copy image
            shutil.copy2(img_path, os.path.join(dest_img, dest_name + ext))
            stats[our_split]["images"] += 1

            # Copy and remap label
            label_name = stem + ".txt"
            label_path = os.path.join(lbl_dir, label_name) if lbl_dir else ""
            dest_label = os.path.join(dest_lbl, dest_name + ".txt")

            if label_path and os.path.isfile(label_path):
                shutil.copy2(label_path, dest_label)
                kept, dropped = remap_label_file(dest_label, class_mapping)
                stats[our_split]["labels_kept"] += kept
                stats[our_split]["labels_dropped"] += dropped
            else:
                # Create empty label file (background/negative image)
                with open(dest_label, "w") as f:
                    pass

    return stats


def update_dataset_yaml():
    """Update training/dataset.yaml to match our 2-class pipeline."""
    yaml_path = os.path.join(TRAINING_DIR, "dataset.yaml")
    content = """# YOLOv8 dataset config for FTC DECODE ball detection
# 2 classes: green_ball (0), purple_ball (1)
#
# Directory structure:
#   training/
#     images/train/   <- training images
#     images/val/     <- validation images
#     labels/train/   <- YOLO format labels (.txt)
#     labels/val/     <- YOLO format labels (.txt)
#
# Label format (one line per object):
#   class_id  x_center  y_center  width  height
#   (all values normalized 0-1 relative to image dimensions)
#
# Example label file content:
#   0 0.45 0.62 0.03 0.04
#   1 0.52 0.61 0.03 0.04

path: .  # resolved relative to train.py working directory
train: images/train
val: images/val

nc: 2
names:
  0: green_ball
  1: purple_ball
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Updated {yaml_path}")


def print_dataset_stats():
    """Print statistics about the current dataset in the pipeline directories."""
    print("\n" + "=" * 50)
    print("  Dataset Statistics")
    print("=" * 50)

    for split in ("train", "val"):
        img_dir = os.path.join(TRAINING_DIR, "images", split)
        lbl_dir = os.path.join(TRAINING_DIR, "labels", split)

        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(glob.glob(os.path.join(img_dir, ext)))
            images.extend(glob.glob(os.path.join(img_dir, ext.upper())))

        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

        # Count annotations per class
        class_counts = {0: 0, 1: 0}
        empty_labels = 0
        for lbl_path in labels:
            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                empty_labels += 1
                continue
            for line in lines:
                cls_id = int(line.split()[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        print(f"\n  {split}:")
        print(f"    Images:          {len(images)}")
        print(f"    Label files:     {len(labels)}")
        print(f"    Empty labels:    {empty_labels} (background/negative)")
        for cls_id, name in OUR_CLASSES.items():
            print(f"    {name} (cls {cls_id}): {class_counts.get(cls_id, 0)} annotations")

    # Check for unlabeled images
    unlabeled_dir = os.path.join(TRAINING_DIR, "images", "unlabeled")
    if os.path.isdir(unlabeled_dir):
        unlabeled = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            unlabeled.extend(glob.glob(os.path.join(unlabeled_dir, ext)))
            unlabeled.extend(glob.glob(os.path.join(unlabeled_dir, ext.upper())))
        print(f"\n  unlabeled:")
        print(f"    Frames:          {len(unlabeled)}")

    print()


def extract_video_frames(video_path, fps=2):
    """Extract frames from a video file for manual labeling."""
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python not installed.")
        print("  pip install opencv-python")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    dest_dir = os.path.join(TRAINING_DIR, "images", "unlabeled")
    os.makedirs(dest_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  FPS: {video_fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")
    print(f"  Extracting at {fps} fps -> ~{int(duration * fps)} frames")

    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    if frame_interval < 1:
        frame_interval = 1

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    count = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            filename = f"{video_stem}_frame{frame_num:06d}.jpg"
            cv2.imwrite(os.path.join(dest_dir, filename), frame)
            count += 1
            if count % 50 == 0:
                print(f"  Extracted {count} frames...")
        frame_num += 1

    cap.release()
    print(f"  Done: {count} frames saved to {dest_dir}")
    return count


def download_source(source_name, api_key):
    """Download a single dataset source and copy into pipeline dirs."""
    if source_name not in DATASETS:
        print(f"Error: Unknown source '{source_name}'. Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    workspace, project, version = DATASETS[source_name]
    prefix = f"{source_name}_" if source_name != "primary" else ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_dir = os.path.join(tmp_dir, "dataset")
        download_dataset(workspace, project, version, api_key, download_dir)

        # Detect class mapping from downloaded data
        class_mapping = detect_class_mapping(download_dir)
        print(f"  Class mapping: {class_mapping}")

        # Find splits in downloaded data
        splits = find_split_dirs(download_dir)
        if not splits:
            print(f"Error: Could not find image directories in {download_dir}")
            print("  Expected structure: train/images/, valid/images/")
            sys.exit(1)

        print(f"  Found splits: {list(splits.keys())}")

        # Copy into pipeline
        stats = copy_files_to_pipeline(splits, class_mapping, prefix=prefix)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download FTC DECODE training data from Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--source", type=str, default="primary",
                        choices=["primary", "solarflare", "all"],
                        help="Dataset source to download (default: primary)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file to extract frames from")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second to extract from video (default: 2)")
    parser.add_argument("--stats", action="store_true",
                        help="Print current dataset statistics and exit")
    parser.add_argument("--import-zip", type=str, default=None, metavar="PATH",
                        help="Import a manually downloaded Roboflow YOLOv8 zip file")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing images/labels before downloading")
    args = parser.parse_args()

    # Stats-only mode
    if args.stats:
        print_dataset_stats()
        return

    # Video extraction mode
    if args.video:
        count = extract_video_frames(args.video, fps=args.fps)
        print(f"\nExtracted {count} frames for manual labeling.")
        print("Next steps:")
        print("  1. Upload images/unlabeled/ to Roboflow for annotation")
        print("  2. Export annotated dataset and re-run this script to download")
        return

    # Import zip mode — for datasets downloaded via Roboflow web UI
    if args.import_zip:
        import zipfile
        zip_path = args.import_zip
        if not os.path.isfile(zip_path):
            print(f"Error: File not found: {zip_path}")
            sys.exit(1)

        extract_dir = tempfile.mkdtemp(prefix="roboflow_import_")
        print(f"Extracting {zip_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Detect class mapping from the extracted data.yaml
        class_mapping = detect_class_mapping(extract_dir)
        if class_mapping is None:
            print("Warning: Could not detect class mapping, assuming 0=green, 1=purple")
            class_mapping = {0: 0, 1: 1, 2: None}

        # Find train/val splits
        splits = find_split_dirs(extract_dir)
        if not any(v["images"] for v in splits.values()):
            print("Error: No images found in the extracted zip!")
            print(f"  Contents: {os.listdir(extract_dir)}")
            sys.exit(1)

        # Copy into our pipeline
        copy_files_to_pipeline(splits, class_mapping, prefix="import")
        update_dataset_yaml()

        # Cleanup
        shutil.rmtree(extract_dir)

        print("\nImport complete!")
        print_dataset_stats()
        return

    # Dataset download mode
    api_key = get_api_key(args)

    # Optionally clean existing data
    if args.clean:
        for split in ("train", "val"):
            for subdir in ("images", "labels"):
                d = os.path.join(TRAINING_DIR, subdir, split)
                if os.path.isdir(d):
                    shutil.rmtree(d)
                    os.makedirs(d)
                    print(f"Cleaned {d}")

    # Determine which sources to download
    if args.source == "all":
        sources = list(DATASETS.keys())
    else:
        sources = [args.source]

    all_stats = {}
    for source in sources:
        print(f"\n{'='*50}")
        print(f"  Downloading: {source}")
        print(f"{'='*50}")
        stats = download_source(source, api_key)
        all_stats[source] = stats

    # Update dataset.yaml
    update_dataset_yaml()

    # Print summary
    print("\n" + "=" * 50)
    print("  Download Summary")
    print("=" * 50)
    for source, stats in all_stats.items():
        print(f"\n  {source}:")
        for split, s in stats.items():
            if s["images"] > 0:
                print(f"    {split}: {s['images']} images, "
                      f"{s['labels_kept']} annotations kept, "
                      f"{s['labels_dropped']} annotations dropped")

    # Print overall stats
    print_dataset_stats()

    print("Done! You can now train with:")
    print("  python training/train.py --device mps")


if __name__ == "__main__":
    main()
