"""
Capture frames from the live camera feed for YOLO training data.

Saves frames as JPEG images to training/images/train/ (or a custom directory).
Use the ESP32-CAM WiFi stream or a USB webcam as the source.

Usage:
    python training/capture_frames.py --count 200              # ESP32 WiFi (default)
    python training/capture_frames.py --count 200 --usb        # USB webcam
    python training/capture_frames.py --count 50 --interval 2  # 1 frame every 2 seconds
    python training/capture_frames.py --output training/images/val  # Save to val set
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Add project root to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def capture_from_esp32(count, interval, output_dir):
    """Capture frames from ESP32-CAM MJPEG stream."""
    import requests

    print(f"Connecting to ESP32-CAM stream: {config.ESP32_STREAM_URL}")
    print("(Make sure you're connected to the ESP32's WiFi)")

    try:
        stream = requests.get(config.ESP32_STREAM_URL, stream=True, timeout=(5, 30))
    except Exception as e:
        print(f"Error: Could not connect — {e}")
        return 0

    if stream.status_code != 200:
        print(f"Error: Stream returned HTTP {stream.status_code}")
        return 0

    print("Connected! Capturing frames...")
    saved = 0
    buf = b""
    last_save = 0

    try:
        for chunk in stream.iter_content(chunk_size=16384):
            if not chunk:
                continue

            buf += chunk

            while True:
                start = buf.find(b"\xff\xd8")
                if start == -1:
                    buf = buf[-2:] if len(buf) > 2 else buf
                    break

                end = buf.find(b"\xff\xd9", start + 2)
                if end == -1:
                    buf = buf[start:]
                    break

                jpg_bytes = buf[start:end + 2]
                buf = buf[end + 2:]

                now = time.time()
                if now - last_save < interval:
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    continue

                filename = f"frame_{saved:04d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                saved += 1
                last_save = now
                print(f"  [{saved}/{count}] {filename} ({frame.shape[1]}x{frame.shape[0]})")

                if saved >= count:
                    break

            if saved >= count:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stream.close()

    return saved


def capture_from_usb(count, interval, output_dir, cam_index=0):
    """Capture frames from USB webcam."""
    print(f"Opening USB webcam (index {cam_index})...")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {cam_index}")
        return 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened ({w}x{h}). Capturing frames...")

    saved = 0
    try:
        while saved < count:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Read failed, retrying...")
                time.sleep(0.1)
                continue

            filename = f"frame_{saved:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1
            print(f"  [{saved}/{count}] {filename} ({frame.shape[1]}x{frame.shape[0]})")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()

    return saved


def main():
    parser = argparse.ArgumentParser(description="Capture frames for YOLO training")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of frames to capture (default: 100)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between captures (default: 0.5)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: training/images/train)")
    parser.add_argument("--usb", nargs="?", const=0, type=int, default=None,
                        metavar="INDEX",
                        help="Use USB webcam instead of ESP32 (default index: 0)")
    args = parser.parse_args()

    # Default output dir
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "images", "train")

    os.makedirs(output_dir, exist_ok=True)

    # Check existing files
    existing = [f for f in os.listdir(output_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if existing:
        print(f"Note: {len(existing)} images already in {output_dir}")

    print(f"Capturing {args.count} frames (interval: {args.interval}s)")
    print(f"Output: {output_dir}")
    print()

    if args.usb is not None:
        saved = capture_from_usb(args.count, args.interval, output_dir, args.usb)
    else:
        saved = capture_from_esp32(args.count, args.interval, output_dir)

    print()
    print(f"Done! Saved {saved} frames to {output_dir}")
    if saved > 0:
        print()
        print("Next steps:")
        print("  1. Annotate images with labelImg or Roboflow")
        print("     - Class 0: green_ball")
        print("     - Class 1: purple_ball")
        print("     - Export in YOLO format")
        print("  2. Place .txt label files in training/labels/train/")
        print("  3. Move ~20% of images+labels to images/val/ and labels/val/")
        print("  4. Run: python training/train.py")


if __name__ == "__main__":
    main()
