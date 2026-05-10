"""
Visualize the motion-blur augmentation pipeline used in train.py against
real match-footage frames so we can sanity-check that the synthetic
blur looks like the blur the live camera produces.

Usage:
    python training/preview_motion_blur.py
    python training/preview_motion_blur.py --footage match_footage/355_Point_Match.mp4

Output:
    training/preview_aug/
        synthetic_NN.jpg   — original | each augmentation alone | full pipeline
        real_NN.jpg        — frames sampled from match footage (real motion blur)
        index.html         — single page that views them all side-by-side
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
OUT_DIR = os.path.join(TRAINING_DIR, "preview_aug")


def _load_aug_pipeline_individual():
    """Build the SAME individual transforms train.py installs into the
    ultralytics dataloader, but each as its own Compose at p=1.0 so we
    can render a per-transform preview alongside the full pipeline."""
    import albumentations as A
    # Keep these IN SYNC with the production pipeline in train.py
    # (patch_albumentations_for_motion). Mismatched preview = misleading.
    MOTION_BLUR_LIMIT = (11, 41)
    DIRECTION_RANGE = (0.3, 1.0)  # bias toward trailing (asymmetric) blur
    blocks = [
        ("motion_blur_med",   A.MotionBlur(blur_limit=(15, 21),
                                           direction_range=DIRECTION_RANGE,
                                           p=1.0)),
        ("motion_blur_long",  A.MotionBlur(blur_limit=(31, 41),
                                           direction_range=DIRECTION_RANGE,
                                           p=1.0)),
        ("median_blur",       A.MedianBlur(blur_limit=(3, 7), p=1.0)),
        ("image_compression", A.ImageCompression(quality_range=(40, 80), p=1.0)),
        ("clahe",             A.CLAHE(p=1.0)),
    ]
    full = A.Compose([
        A.Blur(p=0.01),
        A.MedianBlur(blur_limit=(3, 7), p=0.10),
        A.MotionBlur(blur_limit=MOTION_BLUR_LIMIT,
                     direction_range=DIRECTION_RANGE, p=1.00),  # forced on
        A.ImageCompression(quality_range=(40, 80), p=0.50),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
    ])
    return [(name, A.Compose([t])) for name, t in blocks] + [("full_pipeline", full)]


def _stack_with_labels(images, labels, label_h=18):
    """Horizontally concat images with a small label strip on top of each."""
    h = max(im.shape[0] for im in images)
    out_imgs = []
    for im, lbl in zip(images, labels):
        # Pad to common height
        if im.shape[0] != h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
            im = np.vstack([im, pad])
        # Label strip
        strip = np.zeros((label_h, im.shape[1], 3), dtype=im.dtype)
        cv2.putText(strip, lbl, (4, label_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out_imgs.append(np.vstack([strip, im]))
    return np.hstack(out_imgs)


def render_synthetic_samples(n_samples=8):
    """Pick N random training images, apply each augmentation alone +
    the full pipeline, and write side-by-side previews."""
    train_img_dir = os.path.join(TRAINING_DIR, "images", "train")
    if not os.path.isdir(train_img_dir):
        print(f"[!] No training images at {train_img_dir}")
        return []
    candidates = [f for f in os.listdir(train_img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not candidates:
        print(f"[!] No image files in {train_img_dir}")
        return []
    sampled = random.sample(candidates, min(n_samples, len(candidates)))
    pipeline = _load_aug_pipeline_individual()

    out_paths = []
    for i, fname in enumerate(sampled):
        path = os.path.join(train_img_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        panels = [img.copy()]
        labels = ["original"]
        for name, compose in pipeline:
            try:
                out = compose(image=rgb)["image"]
                panels.append(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                labels.append(name)
            except Exception as e:
                print(f"[!] {name} failed: {e}")
        # Resize all panels to a common height for clean comparison
        target_h = 220
        resized = []
        for p in panels:
            scale = target_h / p.shape[0]
            new_w = max(1, int(p.shape[1] * scale))
            resized.append(cv2.resize(p, (new_w, target_h),
                                      interpolation=cv2.INTER_AREA))
        strip = _stack_with_labels(resized, labels)
        out_path = os.path.join(OUT_DIR, f"synthetic_{i:02d}.jpg")
        cv2.imwrite(out_path, strip)
        out_paths.append((out_path, fname))
        print(f"  wrote {out_path}")
    return out_paths


def sample_real_motion_frames(footage_path, n_samples=6):
    """Pull evenly-spaced frames from real match footage so the user can
    eyeball them next to the synthetic samples."""
    if not footage_path or not os.path.isfile(footage_path):
        print(f"[!] Footage not found: {footage_path}")
        return []
    cap = cv2.VideoCapture(footage_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"[!] Could not read frames from {footage_path}")
        return []
    step = max(1, total // (n_samples + 1))
    out_paths = []
    for i in range(n_samples):
        frame_idx = step * (i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # Match what the replay pipeline does: downscale to 720p, JPEG round-trip
        # at quality 60 (matches our REPLAY_SIM_JPEG_QUALITY default) so the
        # preview reflects what YOLO actually sees at inference.
        if frame.shape[0] > 720:
            scale = 720.0 / frame.shape[0]
            new_size = (int(frame.shape[1] * scale), 720)
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok2:
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out_path = os.path.join(OUT_DIR, f"real_{i:02d}.jpg")
        cv2.imwrite(out_path, frame)
        out_paths.append((out_path, f"frame {frame_idx}/{total}"))
        print(f"  wrote {out_path}")
    cap.release()
    return out_paths


def write_index_html(synthetic_paths, real_paths):
    """Single-page HTML viewer so you can flip through everything in a browser."""
    rows_synth = "\n".join(
        f'<div class="row"><div class="lbl">{os.path.basename(p)} ({src})</div>'
        f'<img src="{os.path.basename(p)}"></div>'
        for (p, src) in synthetic_paths
    )
    rows_real = "\n".join(
        f'<div class="row"><div class="lbl">{os.path.basename(p)} ({src})</div>'
        f'<img src="{os.path.basename(p)}"></div>'
        for (p, src) in real_paths
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Motion-blur preview</title>
<style>
 body {{ font-family: -apple-system, sans-serif; background: #0d0f1a; color: #ddd; padding: 16px; }}
 h2 {{ border-bottom: 1px solid #2a2f4a; padding-bottom: 4px; }}
 .row {{ margin: 12px 0; }}
 .lbl {{ font-size: 0.75rem; color: #888; margin-bottom: 4px; }}
 img {{ max-width: 100%; display: block; border: 1px solid #2a2f4a; }}
 .legend {{ background: #14182a; padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; }}
</style></head><body>
<h2>Synthetic motion-blur preview (training data)</h2>
<div class="legend">
  Each row: <b>original</b> | each augmentation in isolation | <b>full pipeline</b>
  (with MotionBlur forced on for visibility — at training time it fires p=0.40).
  If the synthetic blur visually matches the "real motion frames" below at similar speeds,
  the augmentation is well-calibrated. If synthetic looks much harsher or much softer than real,
  tune <code>blur_limit</code> in train.py.
</div>
{rows_synth}
<h2 style="margin-top:32px;">Real frames from match footage</h2>
<div class="legend">
  Real-world reference: frames pulled from your match clip, downscaled to 720p and
  re-encoded at JPEG q=60 (mirrors what live MJPEG delivers). Look at fast-moving
  balls — that's the blur the model has to handle at inference.
</div>
{rows_real}
</body></html>
"""
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Preview motion-blur augmentation")
    parser.add_argument("--samples", type=int, default=8,
                        help="Number of synthetic preview samples")
    parser.add_argument("--footage", type=str,
                        default=os.path.join(PROJECT_ROOT, "match_footage",
                                             "355_Point_Match.mp4"),
                        help="Match footage to sample real motion frames from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[+] Writing previews to {OUT_DIR}/")
    print()

    print("[1/3] Synthetic augmentation samples...")
    synth = render_synthetic_samples(args.samples)

    print()
    print(f"[2/3] Real motion-frame samples from {os.path.basename(args.footage)}...")
    real = sample_real_motion_frames(args.footage)

    print()
    print("[3/3] Index HTML...")
    idx = write_index_html(synth, real)
    print(f"  wrote {idx}")
    print()
    print(f"Open {idx} in a browser to view side-by-side.")


if __name__ == "__main__":
    sys.exit(main() or 0)
