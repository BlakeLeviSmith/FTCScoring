# GPU training recipe

End-to-end steps to train `ftc_motion_v2` on a remote GPU box and pull
the resulting weights back. Designed for any CUDA host — GCP, Lambda,
Vast.ai, RunPod, etc. — over plain SSH. No bucket required.

---

## 1. Pick a GPU

For YOLOv11s + motion-blur aug + `multi_scale=True` (range 320–960) at
batch 32, imgsz 640:

| GPU | VRAM | est. time/50 epochs | typical $/hr (cloud) |
|---|---|---|---|
| T4 (16GB) | tight at multi_scale spikes — drop batch to 16 | ~3-4 h | $0.30-0.50 |
| **L4 (24GB)** ★ | comfortable | ~2-3 h | $0.70-1.00 |
| A10 / A10G (24GB) | comfortable | ~1.5-2 h | $1.00-1.50 |
| A100 40GB | overkill but fast | ~30-60 min | $3-4 |

**Recommendation: L4.** Sweet spot for this workload. 24GB VRAM
handles `multi_scale` spikes cleanly, AMP fp16 keeps it fast, and on
GCP it's `g2-standard-8` ≈ $0.70/hr.

---

## 2. Push code + data to the GPU

```bash
# (on local Mac)
GPU_HOST=user@<gpu-ip>           # whatever your SSH alias is
REMOTE_DIR=~/FTCScoring

# 2a. Code via git
ssh $GPU_HOST "git clone https://github.com/<your-org>/<this-repo>.git $REMOTE_DIR"
# (if private, push your local main first OR use deploy keys)

# 2b. Training data via rsync (133 MB, fast over 100Mbps)
rsync -avz --progress \
  training/images/ training/labels/ training/dataset.yaml \
  $GPU_HOST:$REMOTE_DIR/training/

# 2c. (optional) push existing trained weights so you can A/B compare
rsync -avz --progress \
  training/runs/detect/ftc_motion_v1/weights/best.pt \
  $GPU_HOST:$REMOTE_DIR/training/runs/detect/ftc_motion_v1/weights/best.pt
```

The repo's `.gitignore` already excludes images, labels, and runs/
folders, so step 2a brings code only — no need for git LFS.

---

## 3. Set up the env on the GPU

```bash
ssh $GPU_HOST
cd ~/FTCScoring
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# Install torch matching your CUDA version. Most cloud GPUs are CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics albumentations
# Confirm CUDA is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA L4   (or your GPU name)
```

---

## 4. Disable wandb (one-time, just like local)

```bash
python -c "from ultralytics.utils import SETTINGS; SETTINGS.update({'wandb': False}); print('wandb disabled')"
```

---

## 5. Train inside tmux

```bash
tmux new -s ftc
cd ~/FTCScoring && source .venv/bin/activate
python training/train.py --device 0 --epochs 50 --imgsz 640
# detach with Ctrl-b d ; reattach with: tmux attach -t ftc
```

`train.py` auto-detects CUDA when `--device 0` is set and:
- Bumps batch from 16 → 32
- Bumps workers from 6 → 12
- **Enables AMP (fp16 mixed precision)** — usually 1.5–2× faster on CUDA
- Skips the CPU thread/worker patches

You should see this line near startup:
```
[CUDA] Detected GPU(s). Bumping batch 16→32, workers→12, AMP=on
```

If you don't see it, check `--device 0` was passed.

---

## 6. Pull the trained weights back

```bash
# (on local Mac, after training finishes)
rsync -avz --progress \
  $GPU_HOST:~/FTCScoring/training/runs/detect/ftc_motion_v2/ \
  training/runs/detect/ftc_motion_v2/

# Then in app.py / dashboard, swap to the new model via the hot-swap dropdown
# or set config.YOLO_MODEL_PATH to the new best.pt
```

Total weights size is ~10MB (best.pt + last.pt + plots), trivial pull.

---

## 7. Things to verify on the GPU before kicking off the long run

```bash
# Quick 1-epoch sanity run to confirm everything actually works:
python training/train.py --device 0 --epochs 1 --name _sanity
# Should complete in 2-5 min on an L4. If it errors, fix before
# committing to the full 50-epoch run.
```

Common gotchas:
- **CUDA out of memory** → drop `batch` (try 24 or 16). With
  `multi_scale` the 1.5×imgsz batches are the biggest.
- **`No module named 'albumentations'`** → didn't install in the venv.
- **Slow per-epoch** → check `nvidia-smi` while training; if GPU
  utilization is <80%, dataloader is the bottleneck, raise workers.
- **Ultralytics auto-downloads `yolo11s.pt`** to `~/.cache` on first
  run — no manual download needed.
