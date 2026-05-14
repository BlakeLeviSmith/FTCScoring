"""
Minimal Flask app for the HSV blob counter pivot.

Single video source (--replay PATH), single counting line, single
calibrated color. Just enough to validate that "click ball → set line
→ press play → count goes up" actually works on real footage.
"""

import argparse
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

from hsv_counter import HsvBlobCounter

app = Flask(__name__)

# ---------------- shared state ----------------
counter = HsvBlobCounter()

_state_lock = threading.Lock()
_latest_raw = None        # last raw BGR frame from the source
_latest_jpg = None        # last annotated JPEG bytes for /video_feed
_frame_idx = 0
_total_frames = 0
_src_fps = 30.0
_paused = True            # start paused so user can calibrate + draw line
_seek_to = None
_video_path = None
_dirty = True             # re-render needed (calibration / line changed)
# Labeling mode: pure ground-truth collection. Auto-counter is bypassed
# (no HSV mask, no blob detection), playback is slowed by default,
# and the UI strips out everything that isn't a +1 button. The same
# manual_count gets persisted so a normal-mode run can read it back.
_label_mode = False
_playback_speed = 1.0     # 1.0 = real-time, 0.5 = half-speed, etc.


def grab_loop():
    """Read frames from the video file, run counter.step(), publish JPEG."""
    global _latest_raw, _latest_jpg, _frame_idx, _total_frames, _src_fps
    global _seek_to, _dirty

    cap = cv2.VideoCapture(_video_path)
    if not cap.isOpened():
        print(f"[!] Could not open {_video_path}")
        return
    _total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    base_dt = 1.0 / max(1.0, _src_fps)
    mode = "LABEL" if _label_mode else "NORMAL"
    print(f"[+] Replay [{mode}]: {_video_path}")
    print(f"    {_total_frames} frames @ {_src_fps:.1f} fps "
          f"(playback {_playback_speed:.2f}x)")
    if _label_mode:
        print(f"    Auto-detection DISABLED. Click +1 each time you see a ball cross.")
    else:
        print(f"    [PAUSED] Calibrate a ball + draw a line, then press Play.")

    # Read first frame so the user has something to calibrate against
    ok, frame = cap.read()
    if ok:
        with _state_lock:
            _latest_raw = frame
            _frame_idx = 1
            _dirty = True

    while True:
        # Handle seek
        if _seek_to is not None:
            target = _seek_to
            _seek_to = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if ok:
                with _state_lock:
                    _latest_raw = frame
                    _frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    _dirty = True

        if _paused:
            if _dirty and _latest_raw is not None:
                annotated = counter.step(_latest_raw,
                                         advance_state=False,
                                         label_mode=_label_mode)
                ok2, buf = cv2.imencode(".jpg", annotated,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok2:
                    with _state_lock:
                        _latest_jpg = buf.tobytes()
                        _dirty = False
            time.sleep(0.05)
            continue

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        annotated = counter.step(frame, advance_state=True,
                                 label_mode=_label_mode)
        ok2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            continue
        with _state_lock:
            _latest_raw = frame
            _frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            _latest_jpg = buf.tobytes()
            _dirty = False
        # Speed multiplier: 0.5x = double the wait between frames.
        time.sleep(base_dt / max(0.05, _playback_speed))


# ---------------- routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        last_sent = None
        while True:
            with _state_lock:
                jpg = _latest_jpg
            if jpg is None or jpg is last_sent:
                time.sleep(0.03)
                continue
            last_sent = jpg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   jpg + b"\r\n")
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    since = int(request.args.get("since_cross_seq", 0))
    with _state_lock:
        return jsonify({
            **counter.get_state(since_cross_seq=since),
            "frame": _frame_idx,
            "total_frames": _total_frames,
            "paused": _paused,
            "src_fps": _src_fps,
            "current_video": os.path.abspath(_video_path) if _video_path else None,
            "label_mode": _label_mode,
            "playback_speed": _playback_speed,
        })


@app.route("/api/set_speed", methods=["POST"])
def api_set_speed():
    global _playback_speed
    s = float(request.get_json()["speed"])
    _playback_speed = max(0.05, min(4.0, s))
    return jsonify({"ok": True, "playback_speed": _playback_speed})


@app.route("/api/set_tolerance", methods=["POST"])
def api_set_tolerance():
    global _dirty
    data = request.get_json()
    counter.set_tolerance(int(data["dh"]), int(data["ds"]), int(data["dv"]))
    _dirty = True
    return jsonify({"ok": True, "hsv_tol": list(counter.hsv_tol)})


@app.route("/api/set_view_mode", methods=["POST"])
def api_set_view_mode():
    global _dirty
    counter.set_view_mode(request.get_json()["mode"])
    _dirty = True
    return jsonify({"ok": True, "view_mode": counter.view_mode})


@app.route("/api/set_close_radius", methods=["POST"])
def api_set_close_radius():
    global _dirty
    counter.set_close_radius(int(request.get_json()["r"]))
    _dirty = True
    return jsonify({"ok": True, "mask_close_radius": counter.mask_close_radius})


@app.route("/api/set_aspect", methods=["POST"])
def api_set_aspect():
    global _dirty
    d = request.get_json()
    counter.set_aspect_filter(float(d["lo"]), float(d["hi"]))
    _dirty = True
    return jsonify({"ok": True,
                    "aspect_min": counter.aspect_min,
                    "aspect_max": counter.aspect_max})


@app.route("/api/set_cooldown", methods=["POST"])
def api_set_cooldown():
    d = request.get_json()
    counter.set_cooldown(int(d["frames"]), float(d["radius_mult"]))
    return jsonify({"ok": True,
                    "cooldown_frames": counter.cooldown_frames,
                    "cooldown_radius_mult": counter.cooldown_radius_mult})


@app.route("/api/play_pause", methods=["POST"])
def api_play_pause():
    global _paused
    _paused = not _paused
    return jsonify({"paused": _paused})


@app.route("/api/seek", methods=["POST"])
def api_seek():
    global _seek_to
    f = int(request.get_json().get("frame", 0))
    _seek_to = max(0, min(max(0, _total_frames - 1), f))
    return jsonify({"ok": True, "seek_to": _seek_to})


@app.route("/api/calibrate_color", methods=["POST"])
def api_calibrate_color():
    """Calibrate a color from a user-drawn circle on the current frame.

    Body: {color: "purple"|"green", x, y, r, set_size: bool}
    HSV mean is sampled from inside the circle. If set_size=True the
    circle's area (πr²) is recorded as the per-ball reference."""
    global _dirty
    data = request.get_json()
    color = data["color"]
    x = int(data["x"])
    y = int(data["y"])
    r = int(data["r"])
    set_size = bool(data.get("set_size", False))
    with _state_lock:
        if _latest_raw is None:
            return jsonify({"ok": False, "reason": "no frame yet"}), 400
        frame = _latest_raw.copy()
    ok, info = counter.calibrate_color(frame, color, x, y, r, set_size=set_size)
    _dirty = True
    return jsonify({"ok": ok, **info})


@app.route("/api/line", methods=["POST"])
def api_line():
    global _dirty
    data = request.get_json()
    which = data["which"]      # "classified" | "overflow"
    pts = data["points"]
    ok = counter.set_line(which, tuple(pts[0]), tuple(pts[1]))
    _dirty = True
    return jsonify({"ok": ok, "which": which})


@app.route("/api/manual_increment", methods=["POST"])
def api_manual_increment():
    """Ground-truth counter increment/decrement. Records the current
    video frame so we can reconstruct a timeline of when each ball was
    seen by the human scorer."""
    data = request.get_json()
    with _state_lock:
        frame = _frame_idx
    ok = counter.increment_manual(
        data["line"], data["color"], int(data.get("delta", 1)),
        frame_idx=frame)
    return jsonify({"ok": ok, "manual_count": counter.manual_count})


@app.route("/api/reset_manual", methods=["POST"])
def api_reset_manual():
    counter.reset_manual()
    return jsonify({"ok": True})


@app.route("/api/reset_count", methods=["POST"])
def api_reset_count():
    global _dirty
    counter.reset_count()
    _dirty = True
    return jsonify({"ok": True, "count": 0})


@app.route("/api/clear_calibration", methods=["POST"])
def api_clear_calibration():
    global _dirty
    from hsv_counter import COLORS, LINES
    for c in COLORS:
        counter.hsv[c] = None
    counter.ball_area_px = None
    for n in LINES:
        counter.lines[n] = None
    counter.reset_count()
    counter.reset_manual()
    counter._save()
    _dirty = True
    return jsonify({"ok": True})


# ---------------- main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True,
                        help="Path to a video file to play.")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--label", action="store_true",
                        help="Labeling mode: skip auto-detection, slow "
                             "playback, simplified UI for ground-truth "
                             "click-counting.")
    parser.add_argument("--speed", type=float, default=None,
                        help="Playback speed multiplier (default 1.0 in "
                             "normal mode, 0.5 in label mode).")
    args = parser.parse_args()

    if not os.path.exists(args.replay):
        raise SystemExit(f"Video not found: {args.replay}")

    _video_path = args.replay
    _label_mode = bool(args.label)
    _playback_speed = (args.speed if args.speed is not None
                       else (0.5 if _label_mode else 1.0))
    # Tag any existing manual ground-truth with this video. If the user
    # is opening a DIFFERENT clip than the one the ground-truth was
    # collected for, the UI will show a warning so they don't compare
    # apples to oranges.
    saved_src = getattr(counter, "manual_source_video", None)
    if saved_src is None:
        # First time — record this as the source.
        counter.tag_manual_source(os.path.abspath(_video_path))
    elif os.path.abspath(_video_path) != saved_src:
        print(f"[!] manual ground-truth was recorded against:")
        print(f"      {saved_src}")
        print(f"    but you opened:")
        print(f"      {os.path.abspath(_video_path)}")
        print(f"    Reset manual (UI button) before scoring this clip.")
    threading.Thread(target=grab_loop, daemon=True).start()

    print("=" * 60)
    print(f"  HSV Ball Counter — {'LABEL MODE' if _label_mode else 'normal mode'}")
    print("=" * 60)
    print(f"  Web UI:   http://localhost:{args.port}")
    print(f"  Source:   {args.replay}")
    print(f"  Playback: {_playback_speed:.2f}x")
    if _label_mode:
        print(f"  --label is set: auto-counter is OFF, just record ground truth.")
    print("=" * 60)
    app.run(host=args.host, port=args.port, threaded=True,
            debug=False, use_reloader=False)
