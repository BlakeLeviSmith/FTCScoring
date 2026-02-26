"""
Train YOLOv8 nano on FTC DECODE ball detection dataset.

Usage:
    python training/train.py                     # Train with defaults
    python training/train.py --epochs 200        # More epochs
    python training/train.py --resume            # Resume interrupted training
    python training/train.py --device mps        # Use Apple Silicon GPU

Output:
    training/runs/detect/ftc_balls/weights/best.pt
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for FTC ball detection")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, reduce if OOM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640 — matches VGA)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu, mps (Apple Silicon), 0 (CUDA GPU)")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Pretrained weights (default: yolov8n.pt)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted training")
    parser.add_argument("--name", default="ftc_balls",
                        help="Run name (default: ftc_balls)")
    args = parser.parse_args()

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
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
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
    print(f"    python app.py --yolo --yolo-model {best_pt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
