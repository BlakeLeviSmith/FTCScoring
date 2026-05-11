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
    MOTION_BLUR_LIMIT = (21, 71)
    DIRECTION_RANGE = (0.3, 1.0)  # bias toward trailing (asymmetric) blur
    blocks = [
        # Two MotionBlur previews at different streak lengths so you
        # can see the spread of intensities the trainer samples from.
        ("motion_blur_med",   A.MotionBlur(blur_limit=(25, 35),
                                           direction_range=DIRECTION_RANGE,
                                           p=1.0)),
        ("motion_blur_long",  A.MotionBlur(blur_limit=(55, 71),
                                           direction_range=DIRECTION_RANGE,
                                           p=1.0)),
        # NEW for v3: scale-down preview. Shows what an aggressively
        # downscaled training image looks like — this is what the
        # trainer's `scale=0.7` produces (~0.3-0.5× scaling). Compare
        # the apparent ball size here to a real ball in your match
        # footage to verify the scale matches deployment.
        ("scale_down_0.4x", _scale_down_preview(0.4)),
        ("scale_down_0.6x", _scale_down_preview(0.6)),
        ("median_blur",       A.MedianBlur(blur_limit=(3, 7), p=1.0)),
        ("image_compression", A.ImageCompression(quality_range=(40, 80), p=1.0)),
    ]
    # FULL PIPELINE — combines everything albumentations does PLUS the
    # scale-down (since ultralytics' `scale=0.7` would do something
    # similar to ~half the training images). MotionBlur and
    # ImageCompression both forced on with full intensity so the
    # preview shows the WORST-case stacked transformation a training
    # image might see.
    #
    # NOTE: this still doesn't show ultralytics' MOSAIC (4-image
    # collages) or COPY_PASTE (ball instances grafted from one image
    # onto another). Those happen INSIDE the ultralytics dataloader,
    # downstream of our albumentations patch. To see those samples,
    # ultralytics auto-saves train_batch0.jpg / train_batch1.jpg /
    # train_batch2.jpg to runs/detect/{name}/ at the start of any
    # training run. Run a quick `python training/train.py --epochs 1`
    # locally (or wait for the L4 run to start) to see real cluster
    # samples before the full retrain commits.
    full_pipeline_with_scale = A.Compose([
        # Per-image scale-down BEFORE blur, so the blur applies to the
        # already-shrunk content (mimics what training sees).
        _ShrinkAndPadCls(0.5, p=1.0),
        A.MotionBlur(blur_limit=MOTION_BLUR_LIMIT,
                     direction_range=DIRECTION_RANGE, p=1.00),
        A.ImageCompression(quality_range=(40, 80), p=1.00),
    ])
    return [(name, A.Compose([t])) if not isinstance(t, A.Compose) else (name, t)
            for name, t in blocks] + [
                ("full_pipeline (scale+blur+compress, ULTRALYTICS mosaic/copy_paste NOT shown)",
                 full_pipeline_with_scale),
            ]


# Top-level so the full_pipeline Compose can reference it.
import albumentations as _A_for_class
class _ShrinkAndPadCls(_A_for_class.ImageOnlyTransform):
    def __init__(self, scale, p=1.0):
        super().__init__(p=p)
        self.scale = scale
    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.scale), int(w * self.scale)
        small = cv2.resize(img, (new_w, new_h),
                           interpolation=cv2.INTER_AREA)
        canvas = np.zeros_like(img)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = small
        return canvas
    def get_transform_init_args_names(self):
        return ("scale",)


def _scale_down_preview(scale_factor):
    """Compose that downscales the image to scale_factor of its size and
    pads back to the original with black. Mimics what ultralytics' built-in
    `scale=0.7` augmentation does to many training images — shrinks the
    content so apparent ball size matches the smaller-on-screen reality
    of a deployment camera that's far from the ramp."""
    import albumentations as A
    return A.Compose([_ShrinkAndPadCls(scale_factor, p=1.0)])


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
<h2>Synthetic augmentation preview (training data)</h2>
<div class="legend">
  Each row: <b>original</b> | each augmentation alone | <b>full_pipeline</b>
  (scale-down + motion blur + JPEG compression all stacked, every transform forced ON
  to show worst-case stacking).
  <br><br>
  <b style="color:#fb8;">⚠ NOT shown here:</b> ultralytics' <code>mosaic=0.7</code>
  (4-image collages — synthesizes dense scenes by combining 4 training images into one)
  and <code>copy_paste=0.5</code> (pastes ball instances from one image onto another —
  synthesizes overlapping clusters). Those operate INSIDE the ultralytics dataloader,
  downstream of our albumentations layer.
  <br><br>
  To see those samples, ultralytics auto-saves
  <code>train_batch0.jpg</code> / <code>train_batch1.jpg</code> /
  <code>train_batch2.jpg</code> to <code>training/runs/detect/&lt;run-name&gt;/</code> at the start
  of any training run. The L4 retrain will dump these in the first ~30 seconds; we'll
  pull them back to inspect before letting the full 50 epochs proceed.
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
