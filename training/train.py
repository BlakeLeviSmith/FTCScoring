"""
Train YOLOv8 nano on FTC DECODE ball detection dataset.

Usage:
    python training/train.py                     # Train with defaults
    python training/train.py --epochs 200        # More epochs
    python training/train.py --resume            # Resume interrupted training
    python training/train.py --device mps        # Use Apple Silicon GPU
    python training/train.py --no-motion-aug     # Disable motion-blur augmentation

Output:
    training/runs/detect/ftc_motion_v1/weights/best.pt
"""

import argparse
import os
import sys

# Disable third-party trainer integrations BEFORE importing ultralytics.
# Each of these auto-attaches a callback during training that tries to
# log runs to its respective service. wandb in particular crashes
# because ultralytics passes the absolute `project` path (with `/`)
# which violates wandb's project-name validator. We don't use any of
# these — keep training local-only.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("COMET_MODE", "disabled")
os.environ.setdefault("MLFLOW_TRACKING_URI", "")
os.environ.setdefault("DVCLIVE_OPEN", "false")


def patch_trainer_for_cpu_speed(target_workers=6, torch_threads=4):
    """Undo ultralytics' CPU-only `args.workers = 0` and cap torch
    intra-op threads. ONLY relevant on CPU/MPS — this is a no-op on CUDA
    (we never enter the patch branch because workers won't be zeroed)."""
    import torch
    torch.set_num_threads(torch_threads)
    print(f"[FTC-perf] torch.set_num_threads({torch_threads})")

    from ultralytics.engine.trainer import BaseTrainer
    _orig_init = BaseTrainer.__init__

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if self.args.workers == 0 and self.device.type in {"cpu", "mps"}:
            self.args.workers = target_workers
            print(f"[FTC-perf] Restored workers={target_workers} "
                  f"(ultralytics zeroed it out for {self.device.type})")

    BaseTrainer.__init__ = patched_init


def patch_albumentations_for_motion():
    """Replace ultralytics' default Albumentations transform list with one
    tuned for moving-ball detection.

    Ultralytics installs a quiet built-in Albumentations pipeline (Blur,
    MedianBlur, ToGray, CLAHE — each at p=0.01) when the package is
    available. That's not enough motion variation for our use case.
    We monkey-patch its initializer so the dataloader picks up:
      - MotionBlur at high probability (the headline change)
      - MedianBlur for generic mild blur
      - ImageCompression to mimic the live MJPEG stream's artifacts
      - ToGray + CLAHE kept at low p to retain the original behavior

    Returns True if patched, False if albumentations isn't installed.
    """
    try:
        import albumentations as A
    except ImportError:
        print("[WARN] albumentations not installed — motion-blur aug disabled.")
        print("       pip install albumentations")
        return False

    from ultralytics.data import augment as _aug

    def patched_init(self, p=1.0):
        self.p = p
        # All our transforms are pixel-value-only (no spatial geometry
        # changes), so bboxes don't need to be transformed alongside.
        # Ultralytics' Albumentations call site passes only `image`, so
        # adding bbox_params here would error out. Leaving it off mirrors
        # what ultralytics does for its own non-spatial pipeline.
        self.contains_spatial = False
        self.transform = None
        prefix = "[FTC-aug] "
        try:
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(blur_limit=(3, 7), p=0.10),
                # Long, ASYMMETRIC streaks — real footage of fast balls
                # shows trailing motion blur (the ball leaves a streak
                # behind it as it moves). Default direction_range
                # (-1.0, 1.0) is symmetric, which makes balls look like
                # they're vibrating in place. (0.3, 1.0) biases toward
                # trailing blur. blur_limit up to 41 reaches the streak
                # lengths we see on the fastest balls in match footage.
                A.MotionBlur(blur_limit=(11, 41),
                             direction_range=(0.3, 1.0),
                             p=0.50),
                A.ImageCompression(quality_range=(40, 80), p=0.30),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
            ]
            self.transform = A.Compose(T)
            print(f"{prefix}Custom Albumentations pipeline installed:")
            for t in T:
                print(f"  - {type(t).__name__:18s} p={t.p:.2f}")
        except Exception as e:
            print(f"{prefix}init failed ({e}); falling back to no-op transform.")
            self.transform = None

    _aug.Albumentations.__init__ = patched_init
    return True


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for FTC ball detection")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, reduce if OOM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu, mps (Apple Silicon), 0 (CUDA GPU)")
    parser.add_argument("--weights", default="yolo11s.pt",
                        help="Pretrained weights (default: yolo11s.pt — newer "
                             "architecture, ~3× params of yolov8n)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted training")
    parser.add_argument("--name", default="ftc_motion_v2",
                        help="Run name (default: ftc_motion_v2)")
    parser.add_argument("--no-motion-aug", action="store_true",
                        help="Disable the custom motion-blur augmentation pipeline")
    parser.add_argument("--no-multi-scale", action="store_true",
                        help="Disable multi-scale training (otherwise on by default — "
                             "trains at 0.5×–1.5× imgsz random per batch, "
                             "improves robustness to ball pixel size at inference)")
    args = parser.parse_args()

    # Install the augmentation patch BEFORE any ultralytics dataloader is
    # constructed. Once model.train() is called the patched class is what
    # the dataloader picks up.
    if not args.no_motion_aug:
        patch_albumentations_for_motion()
    else:
        print("[--no-motion-aug] Skipping motion-blur augmentation patch.")

    # Speed-tune for CPU only. On CUDA the bottleneck moves to the
    # GPU and we want to leave torch threading + worker count to
    # ultralytics' GPU defaults.
    if args.device == "cpu":
        patch_trainer_for_cpu_speed(target_workers=6, torch_threads=4)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        print("Install with:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install ultralytics")
        sys.exit(1)

    # Resolve dataset.yaml path relative to this script
    training_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.join(training_dir, "dataset.yaml")

    # Patch dataset.yaml path to absolute so ultralytics resolves correctly
    import yaml
    with open(dataset_yaml, "r") as f:
        ds_config = yaml.safe_load(f)
    ds_config["path"] = training_dir
    patched_yaml = os.path.join(training_dir, "dataset_resolved.yaml")
    with open(patched_yaml, "w") as f:
        yaml.dump(ds_config, f, default_flow_style=False)
    dataset_yaml = patched_yaml

    if not os.path.exists(dataset_yaml):
        print(f"Error: {dataset_yaml} not found")
        sys.exit(1)

    # Check for training images
    train_img_dir = os.path.join(training_dir, "images", "train")
    n_train = len([f for f in os.listdir(train_img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if n_train == 0:
        print(f"Error: No training images found in {train_img_dir}")
        print("1. Run: python training/capture_frames.py --count 200")
        print("2. Annotate with labelImg or Roboflow")
        print("3. Place images in training/images/train/ and labels in training/labels/train/")
        sys.exit(1)

    val_img_dir = os.path.join(training_dir, "images", "val")
    n_val = len([f for f in os.listdir(val_img_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    print(f"Dataset: {n_train} training images, {n_val} validation images")
    print(f"Config:  epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"Device:  {args.device}")
    print(f"Weights: {args.weights}")
    print()

    # ---- CUDA detection + GPU-friendly defaults ----
    # If user passed --device cpu we keep the CPU optimizations above and
    # never touch the GPU. If they pass --device 0 (or auto-detect finds
    # a GPU), bump batch / workers / enable AMP so the GPU isn't starved.
    is_cuda = False
    try:
        import torch as _torch
        is_cuda = (args.device != "cpu") and _torch.cuda.is_available()
    except Exception:
        is_cuda = False

    use_batch = args.batch
    use_workers = 8
    use_amp = False  # CPU/MPS off (MPS tensor index crash)
    if is_cuda:
        # Scale up: a 24GB GPU comfortably handles batch 32 at imgsz 640
        # for yolo11s. multi_scale spikes to 960 still fit.
        use_batch = max(args.batch, 32)
        use_workers = 12
        use_amp = True  # mixed-precision is a free 1.5–2× speedup on CUDA
        print(f"[CUDA] Detected GPU(s). Bumping batch {args.batch}→{use_batch}, "
              f"workers→{use_workers}, AMP=on")

    if args.resume:
        # Resume from last checkpoint
        last_pt = os.path.join(training_dir, "runs", "detect", args.name, "weights", "last.pt")
        if not os.path.exists(last_pt):
            print(f"Error: No checkpoint found at {last_pt}")
            sys.exit(1)
        model = YOLO(last_pt)
        model.train(resume=True)
    else:
        model = YOLO(args.weights)
        model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            batch=use_batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=use_workers,
            name=args.name,
            project=os.path.join(training_dir, "runs", "detect"),

            # Augmentation tuned for small ball detection on fixed-camera setup
            flipud=0.0,        # No vertical flip (ramp has fixed orientation)
            fliplr=0.5,        # Horizontal flip OK (left/right alliance)
            mosaic=0.3,        # Reduced mosaic (balls are small, mosaic can lose them)
            mixup=0.0,         # No mixup (confuses small object boundaries)
            hsv_h=0.015,       # Slight hue variation
            hsv_s=0.5,         # Moderate saturation variation (lighting changes)
            hsv_v=0.4,         # Moderate value variation
            degrees=5.0,       # Slight rotation (camera might be slightly tilted)
            translate=0.1,     # Small translation
            scale=0.3,         # Moderate scale variation

            # Training params
            patience=20,       # Early stopping patience
            save=True,
            save_period=10,    # Save checkpoint every 10 epochs
            plots=True,        # Generate training plots
            verbose=True,
            amp=use_amp,       # AMP on CUDA (1.5-2× speedup), off CPU/MPS
            # Multi-scale training: each batch sampled at 0.5× to 1.5×
            # of imgsz. Lets the model learn to detect balls across the
            # scale range we'll see at inference time (ROI crops vary in
            # size). Cheap relative to the cost of training a bigger
            # model and stacks well with motion-blur augmentation.
            multi_scale=not args.no_multi_scale,
        )

    # Print results location
    results_dir = os.path.join(training_dir, "runs", "detect", args.name)
    best_pt = os.path.join(results_dir, "weights", "best.pt")
    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Best model: {best_pt}")
    print()
    print("  To use in the scoring system:")
    print(f"    python app.py --yolo-model {best_pt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
