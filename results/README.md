# results/

Auto-counter benchmark output, one JSON per clip. Written by
`/api/benchmark` (live) and `benchmark_all.py` (batch). Compares the
pipeline's tripwire events against `labels/<clip>.json`.

Each file has:

```json
{
  "video": "<clipname>",
  "ts": "2026-05-13 14:32:51",
  "tolerance_frames": 150,        // ± window for matching auto ↔ GT events
  "start_frame": 0,                // skip-counting threshold for this clip
  "totals": {
    "gt": 100, "auto": 81,
    "tp": 79, "fp": 2, "fn": 21,   // event-aligned matching
    "raw_acc": 0.81                // min(gt, auto)/max(gt, auto)
  },
  "cells": {                       // same per (alliance × line) cell
    "red":  {"classified": {...}, "overflow": {...}},
    "blue": {"classified": {...}, "overflow": {...}}
  },
  "matched_full":   [...],         // every GT event with an auto match
  "missed_full":    [...],         // GT events with no auto in window (FN)
  "false_pos_full": [...]          // auto events with no GT in window (FP)
}
```

Aggregate across the four labeled clips (current state):

| | total |
|-|-|
| GT events | 529 |
| Auto events | 384 |
| TP / FP / FN | 359 / 25 / 170 |
| precision / recall / F1 | 93 % / 68 % / 0.79 |

See top-level `README.md` for analysis.
