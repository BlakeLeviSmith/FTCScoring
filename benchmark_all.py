"""
Batch benchmark runner. For every clip in match_footage/ that has a
matching labels/<name>.json (= ground truth recorded), runs the YOLO/
CSRT pipeline end-to-end with --auto-run. Each clip's per-clip ROI
zones (roi_configs/<name>.json) are loaded automatically.

Each run produces results/<name>.json on disk. After all clips finish,
prints a summary table.

Usage:
    python benchmark_all.py
    python benchmark_all.py --tol 150          # custom match tolerance (frames)
    python benchmark_all.py --port 8089        # base port (each clip gets +1)
    python benchmark_all.py --skip-existing    # don't re-run clips that already have results

You can interrupt with Ctrl-C and the next invocation picks up the
remaining clips (use --skip-existing to skip already-done ones).
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def labeled_clips():
    """Return absolute paths of every clip that has a labels/<name>.json."""
    out = []
    for clip in sorted(glob.glob(os.path.join(HERE, "match_footage", "*.mp4"))):
        base = os.path.splitext(os.path.basename(clip))[0]
        gt = os.path.join(HERE, "labels", base + ".json")
        if os.path.exists(gt):
            out.append(clip)
    return out


def run_one(clip, port, tol):
    """Spawn the app with --auto-run for ONE clip; block until it exits."""
    base = os.path.basename(clip)
    name = os.path.splitext(base)[0]
    print(f"\n{'='*64}\n  RUN: {base}\n  port: {port}  tol: ±{tol}f\n{'='*64}")
    cmd = [
        sys.executable, os.path.join(HERE, "app.py"),
        "--replay", clip,
        "--port", str(port),
        "--auto-run",
        "--no-loop",
    ]
    print("  $ " + " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    # Inherit stdout/stderr so the per-run logs are visible live.
    rc = subprocess.call(cmd, cwd=HERE)
    dt = time.time() - t0
    print(f"  exit code: {rc}  ({dt:.1f}s)")
    # Check that results actually got written
    rpath = os.path.join(HERE, "results", name + ".json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            d = json.load(f)
        return d
    return None


def fmt_pct(x):
    return "—" if x is None else f"{x*100:.0f}%"


def print_summary(results):
    print(f"\n{'='*78}\n  BENCHMARK SUMMARY  ({len(results)} clip(s))\n{'='*78}")
    hdr = f"  {'clip':<40s}  {'GT':>4s}  {'Auto':>5s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'raw':>5s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tot_gt = tot_auto = tot_tp = tot_fp = tot_fn = 0
    for name, d in results:
        if d is None:
            print(f"  {name[:40]:<40s}  (no results file)")
            continue
        t = d.get("totals", {})
        gt, au, tp, fp, fn = t.get("gt", 0), t.get("auto", 0), t.get("tp", 0), t.get("fp", 0), t.get("fn", 0)
        tot_gt += gt; tot_auto += au; tot_tp += tp; tot_fp += fp; tot_fn += fn
        print(f"  {name[:40]:<40s}  {gt:>4d}  {au:>5d}  {tp:>4d}  {fp:>4d}  {fn:>4d}  {fmt_pct(t.get('raw_acc')):>5s}")
    if tot_gt or tot_auto:
        raw = min(tot_gt, tot_auto) / max(tot_gt, tot_auto) if max(tot_gt, tot_auto) else None
        print("  " + "-" * (len(hdr) - 2))
        print(f"  {'TOTAL':<40s}  {tot_gt:>4d}  {tot_auto:>5d}  {tot_tp:>4d}  {tot_fp:>4d}  {tot_fn:>4d}  {fmt_pct(raw):>5s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tol", type=int, default=150,
                        help="Match tolerance frames passed to /api/benchmark (default 150 = 5s).")
    parser.add_argument("--port", type=int, default=8089,
                        help="Base port for the per-clip server.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Don't re-run clips that already have a results/<name>.json.")
    args = parser.parse_args()

    clips = labeled_clips()
    if not clips:
        sys.exit("No labeled clips found. Run --label first to record ground truth.")

    print(f"Found {len(clips)} labeled clip(s):")
    for c in clips:
        print(f"  · {os.path.basename(c)}")

    if args.skip_existing:
        kept = []
        for c in clips:
            name = os.path.splitext(os.path.basename(c))[0]
            if os.path.exists(os.path.join(HERE, "results", name + ".json")):
                print(f"  (skip: {name})")
                continue
            kept.append(c)
        clips = kept

    results = []
    for clip in clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        d = run_one(clip, args.port, args.tol)
        results.append((name, d))

    print_summary(results)
