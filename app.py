"""
FTC DECODE Scoring System - Web Application
Flask server that captures ESP32-CAM feed, runs YOLO detection, and serves a dashboard.

Usage:
    python app.py                                  # Connect to ESP32-CAM over WiFi
    python app.py --usb                            # Use USB webcam (index 0)
    python app.py --replay training/test_videos/match.mp4  # Replay video at ESP32 quality
    python app.py --replay match.mp4 --replay-fps 10       # Slower replay
    python app.py --yolo-model path/to/best.pt     # Custom YOLO model path
    python app.py --port 9000                      # Use a different port
"""

import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
import requests as http_requests
from flask import Flask, Response, jsonify, render_template, request

import config
from scorer import ScoreKeeper

app = Flask(__name__)

# ============= Shared State =============
# Frame lock: grab thread writes, process thread reads
frame_lock = threading.Lock()
latest_frame = None       # Decoded numpy frame for processing
latest_raw_jpg = None     # Raw JPEG bytes from ESP32 (served directly to raw feed)
frame_seq = 0

# Output lock: process thread writes, Flask generators read
output_lock = threading.Lock()
latest_processed_jpg = None   # Pre-encoded JPEG bytes of processed frame
latest_red_roi_jpg = None     # Pre-encoded JPEG of cropped+upscaled Red ROI
latest_blue_roi_jpg = None    # Pre-encoded JPEG of cropped+upscaled Blue ROI
latest_balls = []
latest_stable_pattern = ""
latest_raw_pattern = ""
fps_counter = {"grab": 0.0, "process": 0.0}

JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 80]

# Live camera health stats (resolution, FPS, dropped frames). Updated by the
# grab loop (resolution, grab count) and the process loop (dropped count).
camera_stats = {
    "width": 0,
    "height": 0,
    "resolution_label": "unknown",
    "grab_fps": 0.0,
    "process_fps": 0.0,
    "dropped_total": 0,
    "frames_total": 0,
    "last_reported_ts": 0.0,
}
camera_stats_lock = threading.Lock()


def _resolution_label(w, h):
    """Human-friendly label for common ESP32 framesize resolutions."""
    table = {
        (96, 96): "96x96",
        (160, 120): "QQVGA",
        (176, 144): "QCIF",
        (240, 176): "HQVGA",
        (320, 240): "QVGA",
        (400, 296): "CIF",
        (640, 480): "VGA",
        (800, 600): "SVGA",
        (1024, 768): "XGA",
        (1280, 720): "HD",
        (1280, 1024): "SXGA",
        (1600, 1200): "UXGA",
        (1920, 1080): "FHD",
        (2048, 1536): "QXGA",
        (2560, 1440): "QHD",
        (2592, 1944): "5MP",
        (640, 360): "nHD",
    }
    return table.get((w, h), f"{w}x{h}")


def _log_camera_health(tag):
    """Print a one-line snapshot of camera stats (rate-limited)."""
    with camera_stats_lock:
        now = time.time()
        if now - camera_stats["last_reported_ts"] < 3.0:
            return
        camera_stats["last_reported_ts"] = now
        w = camera_stats["width"]
        h = camera_stats["height"]
        label = camera_stats["resolution_label"]
        gfps = camera_stats["grab_fps"]
        pfps = camera_stats["process_fps"]
        dropped = camera_stats["dropped_total"]
        total = camera_stats["frames_total"]
    drop_pct = (dropped / max(total, 1)) * 100.0
    print(f"[CAM/{tag}] {label} {w}x{h} | grab {gfps:4.1f} fps | "
          f"process {pfps:4.1f} fps | dropped {dropped}/{total} "
          f"({drop_pct:.1f}%)")


def _note_new_frame(w, h):
    """Record a newly-grabbed frame's resolution; log if it changed."""
    changed = False
    with camera_stats_lock:
        if camera_stats["width"] != w or camera_stats["height"] != h:
            camera_stats["width"] = w
            camera_stats["height"] = h
            camera_stats["resolution_label"] = _resolution_label(w, h)
            changed = True
        camera_stats["frames_total"] += 1
    if changed:
        print(f"[CAM] resolution -> {_resolution_label(w, h)} ({w}x{h})")

detector = None  # Initialized in __main__ after arg parsing
scorer_red = ScoreKeeper()
scorer_blue = ScoreKeeper()
ramp_tracker_red = None
ramp_tracker_blue = None
csrt_tracker_red = None  # MultiBallTracker, populated when TRACKER_BACKEND="csrt"
csrt_tracker_blue = None

# ============= Match State =============
match_state = {
    "phase": "SETUP",           # SETUP | AUTO | TELEOP | ENDED
    "started_at": None,          # time.time() when current phase started
    "phase_duration": 0,         # seconds for current phase
    "auto_snapshot": {           # Scoring snapshot at end of AUTO
        "red": None, "blue": None
    },
    "final_snapshot": {          # Scoring snapshot at end of TELEOP
        "red": None, "blue": None
    },
}
match_state_lock = threading.Lock()

# ROI-only mode: when True, crop alliance ROIs and upscale before YOLO inference
# roi_only_mode removed — crop+upscale auto-enables when any ROI is set

# ============= ROI Persistence =============
ROI_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "roi_config.json")
ROI_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "roi_configs")
_roi_cache = None  # In-memory cache — avoids disk reads on every frame
# Per-clip ROI override: when --replay is used, this is the path to
# roi_configs/<clipname>.json. Reads/writes go through this file
# instead of the global one, so each clip's camera angle gets its own
# zones. Falls back to global roi_config.json when this is None.
_roi_per_clip_path = None


def _default_roi_config():
    """Return a blank dual-alliance ROI config."""
    return {
        "red": {"roi": None, "gate": None, "exit": None, "divider": None},
        "blue": {"roi": None, "gate": None, "exit": None, "divider": None},
    }


def _active_roi_path():
    """Per-clip path if set, else the global file."""
    return _roi_per_clip_path or ROI_CONFIG_PATH


def load_roi_config():
    """Load ROI config from memory cache, falling back to disk on first call.

    When _roi_per_clip_path is set, that file is the canonical store.
    If it doesn't exist yet, we seed it from the global roi_config.json
    so the user gets reasonable defaults the first time they open a
    new clip — they can then adjust + save and the per-clip file gets
    written without affecting the global one.
    """
    global _roi_cache
    if _roi_cache is not None:
        return _roi_cache
    path = _active_roi_path()
    sources = [path] if path == ROI_CONFIG_PATH else [path, ROI_CONFIG_PATH]
    for src in sources:
        if not os.path.exists(src):
            continue
        try:
            with open(src, "r") as f:
                data = json.load(f)
            if "red" not in data and "blue" not in data:
                # Old single-ROI format → migrate
                migrated = _default_roi_config()
                if data.get("roi") or data.get("gate"):
                    migrated["red"]["roi"] = data.get("roi")
                    migrated["red"]["gate"] = data.get("gate")
                _roi_cache = migrated
            else:
                _roi_cache = data
            return _roi_cache
        except (json.JSONDecodeError, IOError):
            continue
    _roi_cache = _default_roi_config()
    return _roi_cache


def save_roi_config(data):
    """Save ROI to the active path (per-clip or global) + cache."""
    global _roi_cache
    _roi_cache = data
    path = _active_roi_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


STREAM_TIMEOUT = (5, 8)  # (connect, read) — 8s read timeout so crop reconnects faster
RECONNECT_DELAY = 0.3
esp32_configured = False


# ============= Connection Monitor =============
class ConnectionMonitor:
    """Tracks stream health: disconnects, FPS lows, uptime."""

    def __init__(self, report_interval=120):
        self.report_interval = report_interval
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.disconnects = []       # list of (timestamp, duration_seconds)
        self.fps_samples = []       # list of (timestamp, fps)
        self.current_disconnect_start = None
        self.total_frames = 0
        self.lock = threading.Lock()

    def record_disconnect(self):
        with self.lock:
            self.current_disconnect_start = time.time()

    def record_reconnect(self):
        with self.lock:
            if self.current_disconnect_start:
                duration = time.time() - self.current_disconnect_start
                self.disconnects.append((self.current_disconnect_start, duration))
                self.current_disconnect_start = None

    def record_fps(self, fps):
        with self.lock:
            self.fps_samples.append((time.time(), fps))
            # Keep last 30 min of samples
            cutoff = time.time() - 1800
            self.fps_samples = [(t, f) for t, f in self.fps_samples if t > cutoff]

    def record_frame(self):
        with self.lock:
            self.total_frames += 1

    def maybe_print_report(self):
        now = time.time()
        if now - self.last_report_time < self.report_interval:
            return
        self.last_report_time = now
        self._print_report(now)

    def _print_report(self, now):
        with self.lock:
            uptime = now - self.start_time
            mins = int(uptime // 60)
            secs = int(uptime % 60)

            # Disconnects in the last report interval
            cutoff = now - self.report_interval
            recent_dc = [(t, d) for t, d in self.disconnects if t > cutoff]
            all_dc = list(self.disconnects)

            # FPS stats for last interval
            recent_fps = [f for t, f in self.fps_samples if t > cutoff and f > 0]

            print()
            print(f"{'=' * 50}")
            print(f"  CONNECTION REPORT  (uptime: {mins}m {secs}s)")
            print(f"{'=' * 50}")
            print(f"  Total frames:      {self.total_frames}")

            if recent_fps:
                avg_fps = sum(recent_fps) / len(recent_fps)
                min_fps = min(recent_fps)
                max_fps = max(recent_fps)
                print(f"  FPS (last 2min):   avg={avg_fps:.1f}  min={min_fps:.1f}  max={max_fps:.1f}")
            else:
                print(f"  FPS (last 2min):   no data")

            print(f"  Disconnects (last 2min):  {len(recent_dc)}")
            for t, d in recent_dc:
                ts = time.strftime("%H:%M:%S", time.localtime(t))
                print(f"    {ts} — down {d:.1f}s")

            print(f"  Disconnects (total):      {len(all_dc)}")
            if all_dc:
                total_down = sum(d for _, d in all_dc)
                avg_down = total_down / len(all_dc)
                max_down = max(d for _, d in all_dc)
                availability = max(0, (uptime - total_down) / uptime * 100)
                print(f"    Total downtime:  {total_down:.1f}s")
                print(f"    Avg duration:    {avg_down:.1f}s")
                print(f"    Max duration:    {max_down:.1f}s")
                print(f"    Availability:    {availability:.1f}%")
            else:
                print(f"    No disconnects recorded")
            print(f"{'=' * 50}")
            print()


conn_monitor = ConnectionMonitor()


def configure_esp32_camera(force=False):
    """Push framesize + quality to the ESP32 camera.

    Called on every stream (re)connect so that an ESP32 reset (fall, power
    blip) doesn't leave us stuck at the firmware-default framesize/quality.
    The `esp32_configured` flag is informational only when `force=False`;
    callers that want to skip redundant work on the very same connection
    can set it, but the outer reconnect loop clears it.
    """
    global esp32_configured
    if esp32_configured and not force:
        return

    try:
        url = f"{config.ESP32_CONTROL_URL}?var=quality&val={config.ESP32_DEFAULT_QUALITY}"
        resp = http_requests.get(url, timeout=3)
        if resp.status_code == 200:
            print(f"    Set quality={config.ESP32_DEFAULT_QUALITY}")
        else:
            print(f"    [i] /control returned {resp.status_code} — skipping config")
            esp32_configured = True
            return
    except Exception:
        print("    [i] /control not reachable — skipping camera config")
        esp32_configured = True
        return

    if config.ESP32_DEFAULT_FRAMESIZE is not None:
        try:
            url = f"{config.ESP32_CONTROL_URL}?var=framesize&val={config.ESP32_DEFAULT_FRAMESIZE}"
            resp = http_requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"    Set framesize={config.ESP32_DEFAULT_FRAMESIZE}")
                print("    Waiting for camera to stabilize...")
                time.sleep(2)
            else:
                print(f"    [!] framesize returned {resp.status_code}")
        except Exception as e:
            print(f"    [!] Could not set framesize: {e}")
    else:
        print("    Using firmware default framesize")
        time.sleep(1)  # brief stabilization

    # Flip image if configured (upside-down mounted camera)
    if getattr(config, 'ESP32_FLIP_IMAGE', False):
        try:
            http_requests.get(
                f"{config.ESP32_CONTROL_URL}?var=vflip&val=1", timeout=3)
            http_requests.get(
                f"{config.ESP32_CONTROL_URL}?var=hmirror&val=1", timeout=3)
            print("    Set vflip=1 hmirror=1 (image flipped)")
        except Exception:
            print("    [!] Could not set flip/mirror")

    esp32_configured = True


def grab_loop():
    """Fast thread: reads raw MJPEG stream, parses JPEG frames directly."""
    global latest_frame, latest_raw_jpg, frame_seq

    first_connect = True

    while True:
        print(f"[*] Opening MJPEG stream: {config.ESP32_STREAM_URL}")
        stream = None
        try:
            stream = http_requests.get(
                config.ESP32_STREAM_URL, stream=True, timeout=STREAM_TIMEOUT,
            )
            if stream.status_code != 200:
                print(f"[!] Stream returned HTTP {stream.status_code}")
                conn_monitor.record_disconnect()
                time.sleep(3)
                continue
        except Exception as e:
            print(f"[!] Connection failed: {e}")
            print("    1) Is the ESP32-CAM powered on?")
            print("    2) Are you connected to its WiFi network?")
            conn_monitor.record_disconnect()
            time.sleep(3)
            continue

        conn_monitor.record_reconnect()
        print("[+] MJPEG stream connected")

        # Configure camera settings only on first connect. Re-configuring on
        # every reconnect was thrashing the OV5640 (2s stabilization delay +
        # stream tear-down on each cycle). Use /api/camera/reapply to force
        # a reconfigure if the camera settings drifted after a power cycle.
        if first_connect:
            first_connect = False
            try:
                stream.close()
            except Exception:
                pass
            configure_esp32_camera()
        try:
            stream = http_requests.get(
                config.ESP32_STREAM_URL, stream=True, timeout=STREAM_TIMEOUT,
            )
            if stream.status_code != 200:
                print(f"[!] Post-config stream returned HTTP {stream.status_code}")
                time.sleep(3)
                continue
            print("[+] Stream reconnected after config")
        except Exception as e:
            print(f"[!] Post-config reconnect failed: {e}, retrying...")
            time.sleep(3)
            continue

        last_time = time.time()
        grab_count = 0
        first_frame_logged = False
        buf = b""

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

                    frame = cv2.imdecode(
                        np.frombuffer(jpg_bytes, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is None:
                        continue

                    if not first_frame_logged:
                        print(f"    First frame: {frame.shape[1]}x{frame.shape[0]} ({len(jpg_bytes)} bytes)")
                        first_frame_logged = True

                    _note_new_frame(frame.shape[1], frame.shape[0])

                    with frame_lock:
                        latest_frame = frame
                        latest_raw_jpg = jpg_bytes
                        frame_seq += 1

                    grab_count += 1
                    conn_monitor.record_frame()

                # FPS tracking + monitoring
                elapsed = time.time() - last_time
                if elapsed >= 1.0:
                    current_fps = grab_count / elapsed
                    fps_counter["grab"] = current_fps
                    with camera_stats_lock:
                        camera_stats["grab_fps"] = current_fps
                    if grab_count > 0:
                        conn_monitor.record_fps(current_fps)
                    grab_count = 0
                    last_time = time.time()
                    conn_monitor.maybe_print_report()
                    _log_camera_health("ESP32")

        except http_requests.exceptions.ReadTimeout:
            print(f"[!] Stream dead ({STREAM_TIMEOUT[1]}s no data), reconnecting...")
        except Exception as e:
            print(f"[!] Stream error: {e}")

        conn_monitor.record_disconnect()
        try:
            stream.close()
        except Exception:
            pass
        fps_counter["grab"] = 0.0
        time.sleep(RECONNECT_DELAY)


def open_usb_camera(cam_index=0):
    """Open USB webcam on the MAIN thread so macOS can show the camera
    permission dialog. Returns the opened VideoCapture object."""
    import os
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"

    print(f"[*] Opening USB webcam (index {cam_index})...")
    print("    (If prompted, grant camera access to Terminal/iTerm)")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[!] Could not open USB webcam at index {cam_index}")
        print("    Try a different index with: python app.py --usb 1")
        print("    Also check System Settings > Privacy & Security > Camera")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] USB webcam opened — {actual_w}x{actual_h}")
    return cap


def grab_loop_usb(cap):
    """Grab loop for USB webcam."""
    global latest_frame, latest_raw_jpg, frame_seq

    last_time = time.time()
    grab_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] USB webcam read failed, retrying...")
            time.sleep(0.1)
            continue

        _, jpg_buf = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
        jpg_bytes = jpg_buf.tobytes()

        _note_new_frame(frame.shape[1], frame.shape[0])

        with frame_lock:
            latest_frame = frame
            latest_raw_jpg = jpg_bytes
            frame_seq += 1

        grab_count += 1

        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_counter["grab"] = grab_count / elapsed
            with camera_stats_lock:
                camera_stats["grab_fps"] = grab_count / elapsed
            grab_count = 0
            last_time = time.time()
            _log_camera_health("USB")


# ============= Replay State =============
MATCH_FOOTAGE_DIR = os.path.join(os.path.dirname(__file__), "match_footage")
_replay_current_video = None   # Path to current video being replayed
_replay_switch_to = None       # Set by API to trigger a video switch
_replay_target_fps = 30  # default: 30fps = real-time for typical match footage
_replay_paused = True          # Start paused so user can set up zones first
_replay_seek_to = None         # Set by API to trigger a frame seek
_replay_total_frames = 0       # Total frames in current video
_replay_current_frame = 0      # Current frame position
_replay_src_fps = 30           # Source video FPS for time calculations


def grab_loop_replay(video_path, target_fps=15, loop=True):
    """Grab loop that replays a video file, downscaled to live stream quality.

    Frames larger than config.REPLAY_TARGET_HEIGHT are resized down preserving
    aspect ratio so the YOLO model sees the same pixel density it gets from
    the ESP32 stream. Smaller sources are passed through untouched.
    Supports runtime video switching via _replay_switch_to.
    """
    global latest_frame, latest_raw_jpg, frame_seq
    global _replay_current_video, _replay_switch_to, _replay_target_fps
    global _replay_seek_to, _replay_total_frames, _replay_current_frame, _replay_src_fps

    REPLAY_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 70]
    _replay_current_video = video_path
    _replay_target_fps = target_fps

    while True:
        # Check if we should switch to a different video
        if _replay_switch_to is not None:
            video_path = _replay_switch_to
            _replay_switch_to = None
            _replay_current_video = video_path
            print(f"[*] Switching replay to: {os.path.basename(video_path)}")

        frame_delay = 1.0 / _replay_target_fps

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] Cannot open video: {video_path}")
            time.sleep(3)
            if not loop:
                return
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _replay_total_frames = total
        _replay_src_fps = src_fps
        _replay_current_frame = 0

        target_h = getattr(config, "REPLAY_TARGET_HEIGHT", 0) or 0
        if target_h > 0 and src_h > target_h:
            scale = target_h / float(src_h)
            out_w = int(round(src_w * scale))
            out_h = target_h
            def _resize(f):
                return cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            out_w, out_h = src_w, src_h
            def _resize(f):
                return f

        # Read sim-compression settings INSIDE _scale on every frame so
        # toggles from the UI take effect immediately without restart.
        def _scale(f):
            f = _resize(f)
            if getattr(config, "REPLAY_SIMULATE_LIVE_COMPRESSION", False):
                q = int(getattr(config, "REPLAY_SIM_JPEG_QUALITY", 60))
                ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    f = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return f

        print(f"[+] Replay: {os.path.basename(video_path)}")
        print(f"    Source: {src_w}x{src_h} @ {src_fps:.1f}fps, {total} frames")
        sim_on = getattr(config, "REPLAY_SIMULATE_LIVE_COMPRESSION", False)
        comp_note = (f", MJPEG-sim q={int(getattr(config, 'REPLAY_SIM_JPEG_QUALITY', 60))}"
                     if sim_on else "")
        print(f"    Output: {out_w}x{out_h} @ {_replay_target_fps}fps{comp_note}")

        last_time = time.time()
        grab_count = 0

        # Show first frame immediately (even while paused)
        ret, first_frame = cap.read()
        if ret:
            first_frame = _scale(first_frame)
            _replay_current_frame = 1
            _, jpg_buf = cv2.imencode(".jpg", first_frame, REPLAY_QUALITY)
            with frame_lock:
                latest_frame = first_frame
                latest_raw_jpg = jpg_buf.tobytes()
                frame_seq += 1
            print("    [PAUSED] Draw your ROI/Gate zones, then click Start")

        while True:
            # Check for video switch mid-playback
            if _replay_switch_to is not None:
                cap.release()
                break

            # Handle seek request (works whether paused or playing)
            if _replay_seek_to is not None:
                target_frame = max(0, min(_replay_seek_to, total - 1))
                _replay_seek_to = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if ret:
                    frame = _scale(frame)
                    _replay_current_frame = target_frame + 1
                    _, jpg_buf = cv2.imencode(".jpg", frame, REPLAY_QUALITY)
                    with frame_lock:
                        latest_frame = frame
                        latest_raw_jpg = jpg_buf.tobytes()
                        frame_seq += 1
                if _replay_paused:
                    continue

            # While paused, keep serving current frame but don't advance
            if _replay_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break
            frame = _scale(frame)
            _replay_current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Encode as JPEG for the raw feed (already at live quality)
            _, jpg_buf = cv2.imencode(".jpg", frame, REPLAY_QUALITY)
            jpg_bytes = jpg_buf.tobytes()

            _note_new_frame(frame.shape[1], frame.shape[0])

            with frame_lock:
                latest_frame = frame
                latest_raw_jpg = jpg_bytes
                frame_seq += 1

            grab_count += 1

            elapsed = time.time() - last_time
            if elapsed >= 1.0:
                fps_counter["grab"] = grab_count / elapsed
                with camera_stats_lock:
                    camera_stats["grab_fps"] = grab_count / elapsed
                grab_count = 0
                last_time = time.time()
                _log_camera_health("REPLAY")

            time.sleep(frame_delay)

        if _replay_switch_to is not None:
            continue  # Go to top of loop to pick up new video

        cap.release()
        if not loop:
            print("[*] Replay finished (single pass)")
            if _auto_run:
                # No browser → benchmark endpoint never gets polled
                # naturally. Self-poll once via Flask test client so
                # the auto-save side-effect fires + writes the results
                # file, then exit so the batch runner moves on.
                print("[*] --auto-run: triggering final benchmark write...")
                try:
                    with app.test_client() as client:
                        r = client.get("/api/benchmark?tol=150")
                        d = r.get_json()
                        if d and d.get("available"):
                            t = d["totals"]
                            print(f"[*] FINAL  GT={t['gt']} Auto={t['auto']} "
                                  f"TP={t['tp']} FP={t['fp']} FN={t['fn']} "
                                  f"raw={(t['raw_acc'] or 0)*100:.0f}%")
                except Exception as e:
                    print(f"[*] benchmark write failed: {e}")
                time.sleep(0.5)
                os._exit(0)
            while True:
                if _replay_switch_to is not None:
                    break
                time.sleep(0.5)
            continue
        print("[*] Replay looping...")


def grab_loop_capture():
    """Grab loop using /capture endpoint (single JPEG polling)."""
    global latest_frame, latest_raw_jpg, frame_seq

    last_time = time.time()
    grab_count = 0
    first_frame = True

    while True:
        try:
            resp = http_requests.get(config.ESP32_CAPTURE_URL, timeout=5)
            if resp.status_code != 200:
                print(f"[!] /capture returned HTTP {resp.status_code}")
                time.sleep(1)
                continue

            jpg_bytes = resp.content
            frame = cv2.imdecode(
                np.frombuffer(jpg_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                time.sleep(0.05)
                continue

            if first_frame:
                print(f"[+] Capture mode active — {frame.shape[1]}x{frame.shape[0]}")
                first_frame = False

            _note_new_frame(frame.shape[1], frame.shape[0])

            with frame_lock:
                latest_frame = frame
                latest_raw_jpg = jpg_bytes
                frame_seq += 1

            grab_count += 1

        except Exception as e:
            print(f"[!] Capture error: {e}")
            time.sleep(1)
            continue

        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_counter["grab"] = grab_count / elapsed
            with camera_stats_lock:
                camera_stats["grab_fps"] = grab_count / elapsed
            grab_count = 0
            last_time = time.time()
            _log_camera_health("CAPTURE")


# Enable with --enhance-roi (or via /api/camera/enhance). Applied AFTER
# upscaling a ROI crop to 640x640.
#   Cheap stack (~3 ms/crop): cubic interp + unsharp mask + CLAHE
#   DNN-SR stack (+~5-10 ms/crop): FSRCNN 2× applied on the raw crop
#       before resize. Needs opencv-contrib-python + the FSRCNN_x2.pb
#       model file; falls back gracefully if either is missing.
_enhance_roi_enabled = False
_dnn_sr_enabled = False
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# DNN super-resolution state
_sr_model = None            # cv2.dnn_superres.DnnSuperResImpl or None
_sr_scale = 2               # 2× — sweet spot for speed
_sr_max_input_side = 320    # skip SR if the crop is already big enough
SR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SR_MODEL_PATH = os.path.join(SR_MODEL_DIR, "FSRCNN_x2.pb")
SR_MODEL_URL = (
    "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/"
    "models/FSRCNN_x2.pb"
)


def _try_init_dnn_sr():
    """Attempt to initialize FSRCNN 2×. Returns True on success."""
    global _sr_model
    if _sr_model is not None:
        return True
    if not hasattr(cv2, "dnn_superres"):
        print("[SR] cv2.dnn_superres not available — "
              "install opencv-contrib-python to enable DNN super-resolution.")
        return False
    if not os.path.exists(SR_MODEL_PATH):
        print(f"[SR] Model not found at {SR_MODEL_PATH} — attempting download...")
        try:
            os.makedirs(SR_MODEL_DIR, exist_ok=True)
            import urllib.request as _urllib
            _urllib.urlretrieve(SR_MODEL_URL, SR_MODEL_PATH)
            print("[SR] Downloaded FSRCNN_x2.pb")
        except Exception as e:
            print(f"[SR] Download failed: {e}. Place FSRCNN_x2.pb in models/ manually.")
            return False
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(SR_MODEL_PATH)
        sr.setModel("fsrcnn", _sr_scale)
        _sr_model = sr
        print(f"[SR] FSRCNN {_sr_scale}× loaded.")
        return True
    except Exception as e:
        print(f"[SR] Model load failed: {e}")
        return False


def _maybe_sr_upscale(crop):
    """Apply FSRCNN 2× to the crop if enabled and the crop is small enough.

    For crops that are already >= _sr_max_input_side on the long side,
    SR offers diminishing returns vs cost, so we skip it — the cubic
    resize to 640x640 handles those fine.
    """
    if not (_dnn_sr_enabled and _sr_model is not None):
        return crop
    h, w = crop.shape[:2]
    if max(h, w) > _sr_max_input_side:
        return crop
    try:
        return _sr_model.upsample(crop)
    except Exception:
        return crop


def _enhance_image(img):
    """Unsharp mask + CLAHE on luminance. Returns a new BGR image."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.4, sigmaY=1.4)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@app.route("/api/camera/enhance", methods=["GET", "POST"])
def api_camera_enhance():
    """Toggle the ROI enhancement pipeline.

    Fields:
      enabled     — cubic + unsharp + CLAHE stack (cheap).
      dnn_sr      — FSRCNN 2× super-resolution stage (requires
                    opencv-contrib-python + FSRCNN_x2.pb model).

    GET:  returns current state of both flags plus SR availability.
    POST: accepts {"enabled": bool, "dnn_sr": bool}, applies set fields.
    """
    global _enhance_roi_enabled, _dnn_sr_enabled
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "enabled" in data:
            _enhance_roi_enabled = bool(data["enabled"])
        if "dnn_sr" in data:
            want = bool(data["dnn_sr"])
            if want:
                if _try_init_dnn_sr():
                    _dnn_sr_enabled = True
                else:
                    _dnn_sr_enabled = False
            else:
                _dnn_sr_enabled = False
    return jsonify({
        "enabled": _enhance_roi_enabled,
        "dnn_sr": _dnn_sr_enabled,
        "dnn_sr_available": _sr_model is not None,
    })


def _poly_to_pixels(poly, w, h):
    """Convert normalized polygon [[x,y],...] to pixel coords as numpy array."""
    return np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)


def _point_in_poly(px, py, poly_px):
    """Check if point (px, py) is inside a pixel-coords polygon."""
    return cv2.pointPolygonTest(poly_px, (float(px), float(py)), False) >= 0


def _balls_in_polygon(balls, poly, w, h):
    """Filter balls whose center falls within a normalized polygon zone."""
    if poly is None or len(poly) < 3:
        return []
    poly_px = _poly_to_pixels(poly, w, h)
    return [b for b in balls
            if _point_in_poly(b.get("center_x", 0), b.get("center_y", 0), poly_px)]


def _draw_dashed_polygon(img, poly_px, color, thickness=2, dash_len=10):
    """Draw a dashed polygon outline on img."""
    n = len(poly_px)
    for i in range(n):
        sx, sy = int(poly_px[i][0]), int(poly_px[i][1])
        ex, ey = int(poly_px[(i + 1) % n][0]), int(poly_px[(i + 1) % n][1])
        dx, dy = ex - sx, ey - sy
        length = int((dx**2 + dy**2) ** 0.5)
        if length == 0:
            continue
        segments = max(1, length // dash_len)
        for j in range(0, segments, 2):
            t0 = j / segments
            t1 = min((j + 1) / segments, 1.0)
            p0 = (int(sx + dx * t0), int(sy + dy * t0))
            p1 = (int(sx + dx * t1), int(sy + dy * t1))
            cv2.line(img, p0, p1, color, thickness)


def _line_side(px, py, p1, p2):
    """Return sign of which side of line p1-p2 the point (px,py) is on.
    +1, -1, or 0."""
    cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
    if cross > 0: return 1
    if cross < 0: return -1
    return 0


def _process_alliance_roi_crop(frame, roi_poly, tracker, scorer_obj, detector_ref, stream_id="default",
                                alliance_roi_data=None, csrt_tracker=None):
    """Crop ROI bounding box, mask non-ROI pixels black, run YOLO once,
    push detections through the alliance's tripwire counter.

    Tripwire architecture: the user draws gate-trip and overflow-trip
    polygons inside the ROI. The tracker emits a count event on rising
    edges of "ball present in tripwire", with a small lockout window
    so a single physical ball can't be counted twice.
    """
    h, w = frame.shape[:2]
    poly_px = _poly_to_pixels(roi_poly, w, h)

    bx, by, bw, bh = cv2.boundingRect(poly_px)
    bx2, by2 = bx + bw, by + bh
    bx, by = max(0, bx), max(0, by)
    bx2, by2 = min(w, bx2), min(h, by2)
    crop_w, crop_h = max(bx2 - bx, 1), max(by2 - by, 1)

    crop = frame[by:by2, bx:bx2]
    if crop.size == 0:
        return [], "", "", None

    # Native sampling: feed the ROI bounding box at its actual pixel
    # dimensions to YOLO (no forced square resize, no aspect distortion).
    if _enhance_roi_enabled:
        processed = _enhance_image(_maybe_sr_upscale(crop))
    else:
        processed = crop
    proc_h, proc_w = processed.shape[:2]

    # ROI polygon mask: blacken pixels outside the user's polygon so YOLO
    # only sees ramp pixels.
    sx = proc_w / float(crop_w)
    sy = proc_h / float(crop_h)
    roi_in_crop = np.array([
        [int((p[0] * w - bx) * sx), int((p[1] * h - by) * sy)]
        for p in roi_poly
    ], dtype=np.int32)
    roi_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_in_crop], 255)
    masked_frame = cv2.bitwise_and(processed, processed, mask=roi_mask)

    # ---- Single YOLO inference on the masked ROI ----
    try:
        balls, stable_pattern, raw_pattern, _ = detector_ref.detect(
            masked_frame, stream_id=stream_id)
    except TypeError:
        balls, stable_pattern, raw_pattern, _ = detector_ref.detect(masked_frame)

    # ---- CSRT correlation-filter tracking (when enabled) ----
    # The detector returns balls WITHOUT track_id when running in
    # CSRT mode (tracking_enabled=False). Each alliance's
    # MultiBallTracker holds N independent CSRT trackers — it updates
    # them on the masked frame, anchors them to fresh YOLO detections
    # via IoU matching, and returns balls WITH stable track_ids.
    if csrt_tracker is not None:
        balls = csrt_tracker.step(masked_frame, balls)

    # Map detection coordinates from processed-crop space back to the full
    # original frame so tripwires (in normalized full-frame coords) hit-test
    # correctly. Filter to balls actually inside the user-drawn ROI poly.
    scale_x = crop_w / float(proc_w)
    scale_y = crop_h / float(proc_h)
    mapped = []
    for b in balls:
        b["center_x"] = bx + b.get("center_x", 0) * scale_x
        b["center_y"] = by + b.get("center_y", 0) * scale_y
        b["x"] = bx + b.get("x", 0) * scale_x
        b["y"] = by + b.get("y", 0) * scale_y
        b["w"] = b.get("w", 0) * scale_x
        b["h"] = b.get("h", 0) * scale_y
        b["area"] = b["w"] * b["h"]
        if _point_in_poly(b["center_x"], b["center_y"], poly_px):
            mapped.append(b)

    # ---- Tripwire counter update ----
    if tracker is not None:
        # Pass the source-clip frame so event-log timestamps align with
        # the ground-truth labels (which use _replay_current_frame).
        # In live camera mode _replay_current_frame is 0; tripwire
        # falls back to its internal counter.
        src_frame = _replay_current_frame if _replay_current_frame else None
        tracker.update(mapped, (h, w), source_frame=src_frame)

        prev = scorer_obj.get_scores()
        totals = tracker.get_totals()
        # Pattern colors are NOT updated live anymore — pattern_locked
        # gates that. We pass an empty list so live updates are a no-op
        # for ramp_colors, while classified/overflow counts still tick.
        scorer_obj.update(
            [],
            classified_total=totals["classified"],
            overflow_total=totals["overflow"],
        )
        _log_score_deltas(stream_id, prev, scorer_obj.get_scores())

    # ---- Build the windowed visualization (single panel, not vstacked) ----
    view_w, view_h = 640, 360
    view = cv2.resize(masked_frame, (view_w, view_h), interpolation=cv2.INTER_AREA)
    sx_disp = view_w / float(proc_w)
    sy_disp = view_h / float(proc_h)

    # Draw tripwire polygons in crop-space so the user sees where they sit
    a_data = alliance_roi_data or {}
    for trip_key, trip_color in (("gate_trip",      (255, 200,   0)),
                                 ("overflow_trip",  (  0, 200, 255))):
        poly = a_data.get(trip_key)
        if poly and len(poly) >= 3:
            pts = []
            for px_n, py_n in poly:
                ox = px_n * w - bx
                oy = py_n * h - by
                pts.append([int(ox * sx * sx_disp), int(oy * sy * sy_disp)])
            cv2.polylines(view, [np.array(pts, dtype=np.int32)],
                          True, trip_color, 2, cv2.LINE_AA)

    # ---- Draw per-track polyline trails ----
    # Each active track's recent positions get connected with a colored
    # polyline. Same color always = same track id. This is the primary
    # debug aid for "is the tracker dropping mid-crossing?" — a long
    # continuous trail = good. Multiple short trails of different colors
    # for what should be one ball = the tracker swapped/lost the id.
    # Prefer CSRT trails (more accurate, proc-coord space) when available;
    # fall back to TripwireCounter's full-frame-coord trails otherwise.
    try:
        from tripwire_counter import stable_color_for_tid as _trail_color
    except ImportError:
        _trail_color = None
    trails_to_draw = None
    trail_coords_in_proc_space = False
    if csrt_tracker is not None and hasattr(csrt_tracker, "get_trails"):
        trails_to_draw = csrt_tracker.get_trails()
        trail_coords_in_proc_space = True  # CSRT trails are in masked_frame coords
    elif tracker is not None and hasattr(tracker, "get_trails"):
        trails_to_draw = tracker.get_trails()
        trail_coords_in_proc_space = False  # TripwireCounter uses full-frame coords
    if trails_to_draw is not None and _trail_color is not None:
        for tid, points in trails_to_draw.items():
            if len(points) < 2:
                continue
            color = _trail_color(int(tid))
            pts_view = []
            for (x, y, _frame, _c) in points:
                if trail_coords_in_proc_space:
                    cx = int(x * sx_disp)
                    cy = int(y * sy_disp)
                else:
                    cx = int((x - bx) * sx * sx_disp)
                    cy = int((y - by) * sy * sy_disp)
                pts_view.append([cx, cy])
            cv2.polylines(view, [np.array(pts_view, dtype=np.int32)],
                          False, color, 2, cv2.LINE_AA)
            hx, hy = pts_view[-1]
            cv2.putText(view, f"#{int(tid)}", (hx + 4, hy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw detection boxes (with track id if available)
    for b in mapped:
        cx = int((b["x"] - bx) * sx * sx_disp)
        cy = int((b["y"] - by) * sy * sy_disp)
        cwb = int(b["w"] * sx * sx_disp)
        chb = int(b["h"] * sy * sy_disp)
        color = (0, 255, 0) if b.get("color") == "G" else (200, 0, 200)
        cv2.rectangle(view, (cx, cy), (cx + cwb, cy + chb), color, 2)
        tid = b.get("track_id")
        tid_str = f" #{tid}" if tid is not None else ""
        cv2.putText(view, f"{b.get('color','?')}{tid_str} {b.get('confidence',0):.2f}",
                    (cx, max(cy - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return mapped, stable_pattern, raw_pattern, view


_ZONE_COLORS = {
    "roi":     (  0, 255,   0),  # green
    "gate":    (  0, 165, 255),  # orange (BGR)
    "exit":    (  0, 255, 255),  # yellow
    "divider": (  0, 215, 255),  # gold
}


def _draw_roi_feed_zones(annotated, roi_data_alliance, meta):
    """Draw thin zone outlines on a 640x640 enhanced-ROI crop, then a tiny
    color-coded key bar at the bottom so the feed has no alliance labels.

    `meta` carries the bounding-box crop origin and size so we can map
    original-frame coords into the upscaled 640×640 crop space:
        crop_x = (orig_x - bx) * 640 / crop_w
        crop_y = (orig_y - by) * 640 / crop_h
    """
    import numpy as _np
    h_out, w_out = annotated.shape[:2]
    bx, by = meta["bx"], meta["by"]
    crop_w = max(meta["crop_w"], 1)
    crop_h = max(meta["crop_h"], 1)
    sx = w_out / float(crop_w)
    sy = h_out / float(crop_h)

    # Original-frame coords for zones are stored normalized to the FULL
    # original frame dimensions, not to the crop. Caller passes them so we
    # can rescale using the stored full-frame size.
    full_w = meta["full_w"]
    full_h = meta["full_h"]

    def to_crop_px(poly_norm):
        pts = []
        for (xn, yn) in poly_norm:
            ox = xn * full_w
            oy = yn * full_h
            cx = (ox - bx) * sx
            cy = (oy - by) * sy
            pts.append([int(cx), int(cy)])
        return _np.array(pts, dtype=_np.int32)

    has = {"roi": False, "gate": False, "exit": False, "divider": False}

    roi_poly = roi_data_alliance.get("roi")
    if roi_poly and len(roi_poly) >= 3:
        poly_px = to_crop_px(roi_poly)
        cv2.polylines(annotated, [poly_px], True, _ZONE_COLORS["roi"], 1, cv2.LINE_AA)
        has["roi"] = True

    gate_poly = roi_data_alliance.get("gate")
    if gate_poly and len(gate_poly) >= 3:
        poly_px = to_crop_px(gate_poly)
        cv2.polylines(annotated, [poly_px], True, _ZONE_COLORS["gate"], 1, cv2.LINE_AA)
        has["gate"] = True

    exit_poly = roi_data_alliance.get("exit")
    if exit_poly and len(exit_poly) >= 3:
        poly_px = to_crop_px(exit_poly)
        cv2.polylines(annotated, [poly_px], True, _ZONE_COLORS["exit"], 1, cv2.LINE_AA)
        has["exit"] = True

    divider = roi_data_alliance.get("divider")
    if divider and isinstance(divider, list) and len(divider) == 2:
        pts = to_crop_px(divider)
        cv2.line(annotated, tuple(pts[0]), tuple(pts[1]),
                 _ZONE_COLORS["divider"], 1, cv2.LINE_AA)
        has["divider"] = True

    _draw_zone_key(annotated, has)


def _draw_zone_key(annotated, has):
    """Tiny color-coded key bar at the bottom of the ROI crop. No text
    labels — just colored swatches and a minimal legend."""
    h_out, w_out = annotated.shape[:2]
    key_h = 14
    y0 = h_out - key_h
    # Semi-transparent backdrop for legibility
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, y0), (w_out, h_out), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    items = [
        ("ROI",     "roi"),
        ("Gate",    "gate"),
        ("Exit",    "exit"),
        ("Divider", "divider"),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.32
    thick = 1
    x = 6
    for label, key in items:
        color = _ZONE_COLORS[key]
        set_ = has.get(key, False)
        swatch_color = color if set_ else (80, 80, 80)
        # 8×8 colored swatch
        cv2.rectangle(annotated, (x, y0 + 3), (x + 8, y0 + 11), swatch_color, -1)
        x += 11
        text_color = (230, 230, 230) if set_ else (120, 120, 120)
        cv2.putText(annotated, label, (x, y0 + 10), font, scale, text_color,
                    thick, cv2.LINE_AA)
        (tw, _th), _ = cv2.getTextSize(label, font, scale, thick)
        x += tw + 10


def _sort_balls_along_ramp(balls, roi_poly, tracker, frame_w, frame_h):
    """Sort balls by their position along the ramp (gate end -> far end).

    Uses the gate zone centroid as the origin and sorts balls by their
    projected distance along the vector from gate to ROI centroid.
    Returns a list of color strings.
    """
    if not balls:
        return []

    # Default: sort by x coordinate (left-to-right) if no gate is set
    gate_poly = getattr(tracker, 'gate_zone', None) if tracker else None

    if gate_poly and len(gate_poly) >= 3:
        # Compute gate centroid
        gx = sum(p[0] for p in gate_poly) / len(gate_poly) * frame_w
        gy = sum(p[1] for p in gate_poly) / len(gate_poly) * frame_h
    else:
        gx = gy = None

    # Compute ROI centroid
    rx = sum(p[0] for p in roi_poly) / len(roi_poly) * frame_w
    ry = sum(p[1] for p in roi_poly) / len(roi_poly) * frame_h

    if gx is None:
        # No gate — sort left-to-right
        sorted_balls = sorted(balls, key=lambda b: b.get("center_x", 0))
    else:
        # Direction vector from ROI centroid toward gate centroid.
        # We want position 1 = farthest from gate (first ball on RAMP),
        # position 9 = closest to gate (most recent ball).
        dx = gx - rx
        dy = gy - ry
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            sorted_balls = sorted(balls, key=lambda b: b.get("center_x", 0))
        else:
            dxn = dx / length
            dyn = dy / length
            # Sort by projection onto the far-end -> gate direction.
            # Smallest projection (farthest from gate) = position 1.
            sorted_balls = sorted(
                balls,
                key=lambda b: (b.get("center_x", 0) - rx) * dxn + (b.get("center_y", 0) - ry) * dyn,
            )

    return [b.get("color", "") for b in sorted_balls if b.get("color") in ("G", "P")]


def _draw_alliance_overlays(processed, roi_data, alliance, color_roi, color_gate):
    """Draw ROI (dashed polygon) and gate (solid polygon) for one alliance."""
    h, w = processed.shape[:2]
    a_data = roi_data.get(alliance, {})

    if a_data.get("roi") and len(a_data["roi"]) >= 3:
        poly_px = _poly_to_pixels(a_data["roi"], w, h)
        _draw_dashed_polygon(processed, poly_px, color_roi, 1)
        for pt in poly_px:
            cv2.circle(processed, (int(pt[0]), int(pt[1])), 2, color_roi, -1)

    if a_data.get("gate") and len(a_data["gate"]) >= 3:
        poly_px = _poly_to_pixels(a_data["gate"], w, h)
        cv2.polylines(processed, [poly_px], True, color_gate, 1)

    if a_data.get("exit") and len(a_data["exit"]) >= 3:
        poly_px = _poly_to_pixels(a_data["exit"], w, h)
        cv2.polylines(processed, [poly_px], True, (0, 255, 255), 1)

    divider = a_data.get("divider")
    if divider and isinstance(divider, list) and len(divider) == 2:
        p1 = (int(divider[0][0] * w), int(divider[0][1] * h))
        p2 = (int(divider[1][0] * w), int(divider[1][1] * h))
        cv2.line(processed, p1, p2, (0, 215, 255), 1, cv2.LINE_AA)


_debug_mode = False   # Set by --debug flag; skips YOLO entirely
_auto_run = False     # Set by --auto-run; starts match + exits at end

# ============= LABEL MODE =============
# Set by --label flag. When True, process_loop bypasses YOLO/CSRT
# entirely and just publishes raw frames with a "LABEL MODE" overlay.
# The user clicks 4 +1 buttons (or hits Q/W/A/S hotkeys) to record
# ground truth. Counts + per-click events persist to labels/<video>.json
# so a benchmarking script can later compare auto-count vs human-count.
_label_mode = False
_label_alliance = "both"   # "red" | "blue" | "both" — locks the active pass
_label_lock = threading.Lock()
# Read-only ground-truth loaded at startup whenever a labels/<clip>.json
# exists for the --replay file. Used by /api/benchmark to compute live
# accuracy of the auto counter vs the human-labeled events.
_ground_truth = None
_label_counts = {
    "red":  {"classified": 0, "overflow": 0},
    "blue": {"classified": 0, "overflow": 0},
}
_label_events = []      # [{seq, frame, t, alliance, line}]
_label_seq = 0
_label_video_basename = None   # set at startup, used to name the JSON

def _label_path():
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "labels")
    os.makedirs(d, exist_ok=True)
    name = (_label_video_basename or "unknown") + ".json"
    return os.path.join(d, name)

def _label_save():
    """Atomic write so concurrent reads never see a half-written file."""
    path = _label_path()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "video": _label_video_basename,
            "counts": _label_counts,
            "events": _label_events,
        }, f, indent=2)
    os.replace(tmp, path)

def _label_load():
    """Load any pre-existing ground truth for THIS video so a labeling
    session can be resumed across restarts."""
    global _label_seq
    path = _label_path()
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            d = json.load(f)
        for a in ("red", "blue"):
            for ln in ("classified", "overflow"):
                _label_counts[a][ln] = int(
                    ((d.get("counts") or {}).get(a) or {}).get(ln, 0))
        _label_events.clear()
        _label_events.extend(d.get("events") or [])
        _label_seq = max((e.get("seq", 0) for e in _label_events), default=0)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass


def process_loop():
    """Processing thread: picks up the latest grabbed frame, runs detection for both alliances."""
    global latest_processed_jpg, latest_red_roi_jpg, latest_blue_roi_jpg
    global latest_balls, latest_stable_pattern, latest_raw_pattern

    last_seq = -1
    last_time = time.time()
    proc_count = 0
    last_idle_print = time.time()

    # Alliance overlay colors (BGR)
    RED_COLOR = (0, 0, 255)
    BLUE_COLOR = (255, 100, 0)
    RED_GATE_COLOR = (0, 100, 255)
    BLUE_GATE_COLOR = (255, 200, 0)

    while True:
        with frame_lock:
            frame = latest_frame
            seq = frame_seq

        if frame is None or seq == last_seq:
            # Heartbeat so the user sees "model alive but no frames"
            # in the terminal. Distinguishes "process_loop is dead"
            # from "no frames are arriving" (replay paused, USB cam
            # detached, ESP32 disconnected, etc).
            now = time.time()
            if now - last_idle_print >= 2.0:
                last_idle_print = now
                reason = "no frame yet" if frame is None else "no new frame (paused?)"
                print(f"[proc/idle] {reason} | seq={seq} last_seq={last_seq}",
                      flush=True)
            time.sleep(0.005)
            continue

        # Dropped = frames that arrived from the grabber while we were busy
        # processing the last one. The grabber always bumps frame_seq, so
        # the gap tells us exactly how many frames the processor skipped.
        if last_seq >= 0:
            skipped = (seq - last_seq) - 1
            if skipped > 0:
                with camera_stats_lock:
                    camera_stats["dropped_total"] += skipped
        last_seq = seq
        roi_data = load_roi_config()
        h, w = frame.shape[:2]

        # ---- LABEL MODE: bypass YOLO/CSRT entirely ----
        # The frame is published as-is with a small overlay; user
        # records ground truth via the +1 buttons / hotkeys. This
        # path is the canonical way to collect benchmark labels.
        if _label_mode:
            processed = frame.copy()
            cv2.rectangle(processed, (8, 8), (220, 44), (0, 0, 0), -1)
            cv2.putText(processed, "LABEL MODE", (14, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
            with _label_lock:
                line1 = (f"R cls={_label_counts['red']['classified']} "
                         f"ovr={_label_counts['red']['overflow']}")
                line2 = (f"B cls={_label_counts['blue']['classified']} "
                         f"ovr={_label_counts['blue']['overflow']}")
            cv2.putText(processed, line1, (14, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
            cv2.putText(processed, line2, (14, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 60), 2)
            _, jpg_buf = cv2.imencode(".jpg", processed, JPEG_ENCODE_PARAMS)
            with frame_lock:
                latest_processed_jpg = jpg_buf.tobytes()
            proc_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                with camera_stats_lock:
                    camera_stats["process_fps"] = proc_count / (now - last_time)
                proc_count = 0
                last_time = now
            continue

        # ---- DEBUG MODE: zone overlays only, no YOLO ----
        if _debug_mode:
            processed = frame.copy()
            # Draw res/fps info
            with camera_stats_lock:
                info = f"{camera_stats['resolution_label']} {camera_stats['width']}x{camera_stats['height']} | {camera_stats['grab_fps']:.1f} fps | Q{config.ESP32_DEFAULT_QUALITY}"
            cv2.putText(processed, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(processed, "DEBUG MODE — no detection", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

            # Draw all alliance overlays
            _draw_alliance_overlays(processed, roi_data, "red", RED_COLOR, RED_GATE_COLOR)
            _draw_alliance_overlays(processed, roi_data, "blue", BLUE_COLOR, BLUE_GATE_COLOR)

            # Encode and publish
            _, proc_buf = cv2.imencode(".jpg", processed, JPEG_ENCODE_PARAMS)
            proc_jpg = proc_buf.tobytes()

            # Generate ROI crop previews (just crops, no YOLO)
            red_roi_jpg = None
            blue_roi_jpg = None
            for alliance in ("red", "blue"):
                a_roi = (roi_data.get(alliance) or {}).get("roi")
                if a_roi and len(a_roi) >= 3:
                    poly_px = _poly_to_pixels(a_roi, w, h)
                    bx, by, bw, bh = cv2.boundingRect(poly_px)
                    bx2, by2 = min(w, bx + bw), min(h, by + bh)
                    bx, by = max(0, bx), max(0, by)
                    crop = frame[by:by2, bx:bx2]
                    if crop.size > 0:
                        resized = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
                        a_data = roi_data.get(alliance) or {}
                        meta = {"bx": bx, "by": by,
                                "crop_w": max(bx2 - bx, 1),
                                "crop_h": max(by2 - by, 1),
                                "full_w": w, "full_h": h}
                        _draw_roi_feed_zones(resized, a_data, meta)
                        _, rbuf = cv2.imencode(".jpg", resized, JPEG_ENCODE_PARAMS)
                        if alliance == "red":
                            red_roi_jpg = rbuf.tobytes()
                        else:
                            blue_roi_jpg = rbuf.tobytes()

            with output_lock:
                latest_processed_jpg = proc_jpg
                latest_red_roi_jpg = red_roi_jpg
                latest_blue_roi_jpg = blue_roi_jpg
                latest_balls = []
                latest_stable_pattern = ""
                latest_raw_pattern = ""

            proc_count += 1
            elapsed = time.time() - last_time
            if elapsed >= 1.0:
                pfps = proc_count / elapsed
                fps_counter["process"] = pfps
                with camera_stats_lock:
                    camera_stats["process_fps"] = pfps
                proc_count = 0
                last_time = time.time()
            continue
        # ---- END DEBUG MODE ----

        all_balls = []
        combined_stable = ""
        combined_raw = ""

        # Auto-detect: use crop+upscale whenever any alliance has an ROI set
        any_roi_set = any(
            (roi_data.get(a) or {}).get("roi")
            for a in ("red", "blue")
        )

        if any_roi_set:
            # --- ROI-cropped YOLO inference (per alliance) ---
            processed = frame.copy()
            annotated_crops = {"red": None, "blue": None}

            for alliance, tracker, scorer_obj, csrt in [
                ("red", ramp_tracker_red, scorer_red, csrt_tracker_red),
                ("blue", ramp_tracker_blue, scorer_blue, csrt_tracker_blue),
            ]:
                a_roi = (roi_data.get(alliance) or {}).get("roi")
                if a_roi is None:
                    scorer_obj.update([])
                    continue

                mapped_balls, s_pat, r_pat, annot_crop = _process_alliance_roi_crop(
                    frame, a_roi, tracker, scorer_obj, detector,
                    stream_id=alliance,
                    alliance_roi_data=roi_data.get(alliance) or {},
                    csrt_tracker=csrt,
                )
                annotated_crops[alliance] = annot_crop
                all_balls.extend(mapped_balls)
                if s_pat:
                    combined_stable = s_pat if not combined_stable else combined_stable
                if r_pat:
                    combined_raw = r_pat if not combined_raw else combined_raw

                # Draw detection boxes colored by alliance
                box_color = RED_COLOR if alliance == "red" else BLUE_COLOR
                for b in mapped_balls:
                    bx = int(b.get("x", 0))
                    by = int(b.get("y", 0))
                    bw = int(b.get("w", 0))
                    bh = int(b.get("h", 0))
                    cv2.rectangle(processed, (bx, by), (bx + bw, by + bh), box_color, 2)
                    label = b.get("color", "?")
                    conf = b.get("confidence", 0)
                    cv2.putText(processed, f"{label} {conf:.2f}",
                                (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, box_color, 1)

        else:
            # --- Full-frame detection, then filter by alliance ROI ---
            balls, stable_pattern, raw_pattern, masks = detector.detect(frame)
            processed = detector.draw_detections(frame, balls, stable_pattern, raw_pattern)
            combined_stable = stable_pattern
            combined_raw = raw_pattern
            all_balls = balls

            for alliance, tracker, scorer_obj in [
                ("red", ramp_tracker_red, scorer_red),
                ("blue", ramp_tracker_blue, scorer_blue),
            ]:
                a_roi = (roi_data.get(alliance) or {}).get("roi")
                if a_roi is not None and len(a_roi) >= 3:
                    alliance_balls = _balls_in_polygon(balls, a_roi, w, h)
                else:
                    alliance_balls = []

                if tracker is not None:
                    src_frame = _replay_current_frame if _replay_current_frame else None
                    tracker.update(alliance_balls, (h, w), source_frame=src_frame)
                if a_roi is not None and len(a_roi) >= 3:
                    ramp_colors = _sort_balls_along_ramp(alliance_balls, a_roi, tracker, w, h)
                else:
                    ramp_colors = [b["color"] for b in alliance_balls if b.get("color") in ("G","P")]
                scorer_obj.update(ramp_colors)

        # Draw alliance overlays (ROI + gate)
        _draw_alliance_overlays(processed, roi_data, "red", RED_COLOR, RED_GATE_COLOR)
        _draw_alliance_overlays(processed, roi_data, "blue", BLUE_COLOR, BLUE_GATE_COLOR)

        # Pre-encode output JPEG
        _, proc_buf = cv2.imencode(".jpg", processed, JPEG_ENCODE_PARAMS)
        proc_jpg = proc_buf.tobytes()

        # Encode ROI crops for live preview feeds (only when ROI-cropped path ran)
        red_roi_jpg = None
        blue_roi_jpg = None
        if any_roi_set:
            red_crop = annotated_crops.get("red")
            if red_crop is not None:
                _, rbuf = cv2.imencode(".jpg", red_crop, JPEG_ENCODE_PARAMS)
                red_roi_jpg = rbuf.tobytes()
            blue_crop = annotated_crops.get("blue")
            if blue_crop is not None:
                _, bbuf = cv2.imencode(".jpg", blue_crop, JPEG_ENCODE_PARAMS)
                blue_roi_jpg = bbuf.tobytes()

        # Publish pre-encoded results
        with output_lock:
            latest_processed_jpg = proc_jpg
            latest_red_roi_jpg = red_roi_jpg
            latest_blue_roi_jpg = blue_roi_jpg
            latest_balls = all_balls
            latest_stable_pattern = combined_stable
            latest_raw_pattern = combined_raw

        proc_count += 1
        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            pfps = proc_count / elapsed
            fps_counter["process"] = pfps
            with camera_stats_lock:
                camera_stats["process_fps"] = pfps
            # Once-per-second visibility into what the CSRT layer is doing.
            # If you see "csrt_red: 0 active" while a match is playing
            # then either the alliance has no ROI drawn or YOLO isn't
            # firing inside it. If active>0 but the dashboard panel
            # still shows 0, the UI poll is broken (open dev tools
            # network tab and check /api/tripwire_debug).
            try:
                r_active = len(csrt_tracker_red._tracks) if csrt_tracker_red else 0
                r_ghost = len(csrt_tracker_red._ghosts) if csrt_tracker_red else 0
                b_active = len(csrt_tracker_blue._tracks) if csrt_tracker_blue else 0
                b_ghost = len(csrt_tracker_blue._ghosts) if csrt_tracker_blue else 0
                n_balls = len(latest_balls or [])
                print(f"[proc] {pfps:5.1f} fps | balls={n_balls:3d} | "
                      f"csrt_red: {r_active} active / {r_ghost} ghosts | "
                      f"csrt_blue: {b_active} active / {b_ghost} ghosts",
                      flush=True)
            except Exception as _e:
                pass
            proc_count = 0
            last_time = time.time()


_blank_jpg = None


def _get_blank_jpg(msg="Waiting for camera..."):
    global _blank_jpg
    if _blank_jpg is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, msg, (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        _, buf = cv2.imencode(".jpg", blank, JPEG_ENCODE_PARAMS)
        _blank_jpg = buf.tobytes()
    return _blank_jpg


_roi_blank_jpg = None
def _get_roi_blank_jpg():
    global _roi_blank_jpg
    if _roi_blank_jpg is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Draw ROI to enable", (140, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        _, buf = cv2.imencode(".jpg", blank, JPEG_ENCODE_PARAMS)
        _roi_blank_jpg = buf.tobytes()
    return _roi_blank_jpg


def generate_mjpeg(feed_type="processed"):
    """Generator that yields pre-encoded JPEG frames."""
    while True:
        jpg_bytes = None

        if feed_type == "raw":
            with frame_lock:
                jpg_bytes = latest_raw_jpg
        elif feed_type == "red_roi":
            with output_lock:
                jpg_bytes = latest_red_roi_jpg
            if jpg_bytes is None:
                jpg_bytes = _get_roi_blank_jpg()
        elif feed_type == "blue_roi":
            with output_lock:
                jpg_bytes = latest_blue_roi_jpg
            if jpg_bytes is None:
                jpg_bytes = _get_roi_blank_jpg()
        else:
            with output_lock:
                jpg_bytes = latest_processed_jpg

        if jpg_bytes is None:
            jpg_bytes = _get_blank_jpg()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")

        time.sleep(0.033)  # ~30fps cap


# ============= Routes =============

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/raw")
def video_feed_raw():
    return Response(generate_mjpeg("raw"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/processed")
def video_feed_processed():
    return Response(generate_mjpeg("processed"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/red_roi")
def video_feed_red_roi():
    return Response(generate_mjpeg("red_roi"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/blue_roi")
def video_feed_blue_roi():
    return Response(generate_mjpeg("blue_roi"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/scores")
def api_scores():
    """Return current scoring data for both alliances as JSON."""
    red_scores = scorer_red.get_scores()
    blue_scores = scorer_blue.get_scores()
    with output_lock:
        red_scores["raw_pattern"] = latest_raw_pattern
        red_scores["stable_pattern"] = latest_stable_pattern
        red_scores["ball_count"] = len(latest_balls)
        blue_scores["raw_pattern"] = latest_raw_pattern
        blue_scores["stable_pattern"] = latest_stable_pattern
        blue_scores["ball_count"] = len(latest_balls)
    return jsonify({
        "red": red_scores,
        "blue": blue_scores,
        "match": _match_state_snapshot(),
        "fps": round(fps_counter.get("process", 0), 1),
        "grab_fps": round(fps_counter.get("grab", 0), 1),
    })


def _match_state_snapshot():
    """Return a JSON-serializable snapshot of the current match state."""
    with match_state_lock:
        phase = match_state["phase"]
        started_at = match_state["started_at"]
        duration = match_state["phase_duration"]
        auto_snap = {
            "red": match_state["auto_snapshot"]["red"],
            "blue": match_state["auto_snapshot"]["blue"],
        }
        final_snap = {
            "red": match_state["final_snapshot"]["red"],
            "blue": match_state["final_snapshot"]["blue"],
        }
    if started_at is not None:
        elapsed = max(0.0, time.time() - started_at)
        remaining = max(0.0, duration - elapsed)
    else:
        elapsed = 0.0
        remaining = 0.0
    return {
        "phase": phase,
        "elapsed": round(elapsed, 2),
        "remaining": round(remaining, 2),
        "phase_duration": duration,
        "auto_snapshot": auto_snap,
        "final_snapshot": final_snap,
    }


@app.route("/api/match/state", methods=["GET"])
def api_match_state():
    """Return current match phase/timer/snapshots."""
    return jsonify(_match_state_snapshot())


@app.route("/api/match/start", methods=["POST"])
def api_match_start():
    """Begin the AUTO phase. Resets trackers/scorers and unpauses replay."""
    global _replay_paused
    if ramp_tracker_red is not None:
        ramp_tracker_red.reset()
    if ramp_tracker_blue is not None:
        ramp_tracker_blue.reset()
    if csrt_tracker_red is not None:
        csrt_tracker_red.reset()
    if csrt_tracker_blue is not None:
        csrt_tracker_blue.reset()
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
    scorer_red.unlock_pattern()
    scorer_blue.unlock_pattern()
    scorer_red.update([])
    scorer_blue.update([])
    with match_state_lock:
        match_state["phase"] = "AUTO"
        match_state["started_at"] = time.time()
        match_state["phase_duration"] = 0  # manual advancement — no timer
        match_state["auto_snapshot"] = {"red": None, "blue": None}
        match_state["final_snapshot"] = {"red": None, "blue": None}
    _replay_paused = False
    print("[MATCH] AUTO started")
    return jsonify({"status": "ok", "match": _match_state_snapshot()})


@app.route("/api/match/reset", methods=["POST"])
def api_match_reset():
    """Full reset: return to SETUP phase, clear snapshots, pause replay."""
    global _replay_paused
    if ramp_tracker_red is not None:
        ramp_tracker_red.reset()
    if ramp_tracker_blue is not None:
        ramp_tracker_blue.reset()
    if csrt_tracker_red is not None:
        csrt_tracker_red.reset()
    if csrt_tracker_blue is not None:
        csrt_tracker_blue.reset()
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
    scorer_red.unlock_pattern()
    scorer_blue.unlock_pattern()
    scorer_red.update([])
    scorer_blue.update([])
    with match_state_lock:
        match_state["phase"] = "SETUP"
        match_state["started_at"] = None
        match_state["phase_duration"] = 0
        match_state["auto_snapshot"] = {"red": None, "blue": None}
        match_state["final_snapshot"] = {"red": None, "blue": None}
    _replay_paused = True
    print("[MATCH] Reset to SETUP")
    return jsonify({"status": "ok", "match": _match_state_snapshot()})


@app.route("/api/match/advance", methods=["POST"])
def api_match_advance():
    """Manually advance the match phase: AUTO -> TELEOP -> ENDED.

    Each call takes a snapshot of the current scores (used for the MOTIF
    pattern scoring at the transition) and advances to the next phase.
    """
    global _replay_paused
    with match_state_lock:
        current = match_state["phase"]

    if current == "AUTO":
        # Snapshot what's on the ramp BEFORE reading the score so the
        # locked pattern is reflected in the AUTO snapshot row.
        _snapshot_and_lock_pattern("AUTO")
        red_snap = scorer_red.get_scores()
        blue_snap = scorer_blue.get_scores()
        if ramp_tracker_red is not None:
            ramp_tracker_red.handoff_phase()
        if ramp_tracker_blue is not None:
            ramp_tracker_blue.handoff_phase()
        with match_state_lock:
            match_state["auto_snapshot"]["red"] = red_snap
            match_state["auto_snapshot"]["blue"] = blue_snap
            match_state["phase"] = "TELEOP"
            match_state["started_at"] = time.time()
        # Pattern stays LOCKED through TELEOP so the AUTO consensus
        # colors remain visible in the UI. They'll be overwritten when
        # TELEOP ends and we sample again.
        _emit_phase_pattern_event("AUTO")
        print("[MATCH] AUTO ended (manual), TELEOP started")
    elif current == "TELEOP":
        _snapshot_and_lock_pattern("TELEOP")
        red_snap = scorer_red.get_scores()
        blue_snap = scorer_blue.get_scores()
        with match_state_lock:
            match_state["final_snapshot"]["red"] = red_snap
            match_state["final_snapshot"]["blue"] = blue_snap
            match_state["phase"] = "ENDED"
            match_state["started_at"] = None
        _replay_paused = True
        _emit_phase_pattern_event("TELEOP")
        print("[MATCH] TELEOP ended (manual), match ENDED")
    else:
        return jsonify({"status": "error",
                        "message": f"Cannot advance from {current}"}), 400

    return jsonify({"status": "ok", "match": _match_state_snapshot()})


@app.route("/api/motif", methods=["POST"])
def api_set_motif():
    """Set the active MOTIF. Optionally specify alliance ('red', 'blue', or both)."""
    data = request.get_json(silent=True) or {}
    motif = data.get("motif", "").upper()
    alliance = data.get("alliance", "").lower()
    if motif not in config.MOTIFS:
        return jsonify({"status": "error", "message": f"Invalid motif: {motif}"}), 400

    if alliance == "red":
        scorer_red.set_motif(motif)
    elif alliance == "blue":
        scorer_blue.set_motif(motif)
    else:
        scorer_red.set_motif(motif)
        scorer_blue.set_motif(motif)
    return jsonify({"status": "ok", "motif": motif, "alliance": alliance or "both"})


@app.route("/api/cam_control", methods=["POST"])
def api_cam_control():
    """Proxy camera settings to the ESP32-CAM's /control endpoint."""
    data = request.get_json(silent=True) or {}
    key = data.get("key", "")
    value = data.get("value", "")
    if not key:
        return jsonify({"status": "error", "message": "Missing key"}), 400

    url = f"{config.ESP32_CONTROL_URL}?var={key}&val={value}"
    try:
        resp = http_requests.get(url, timeout=3)
        return jsonify({"status": "ok", "esp32_response": resp.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 502


@app.route("/api/cam_status", methods=["GET"])
def api_cam_status():
    """Fetch current camera settings from the ESP32-CAM's /status endpoint."""
    url = config.ESP32_STATUS_URL
    try:
        resp = http_requests.get(url, timeout=3)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 502


def _discover_yolo_models():
    """Walk training/runs/detect/*/weights/best.pt and return their paths.
    Used by the UI to populate a "Model" dropdown for hot-swapping."""
    base = os.path.join(os.path.dirname(__file__), "training", "runs", "detect")
    out = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            best = os.path.join(base, name, "weights", "best.pt")
            if os.path.isfile(best):
                # Store relative-to-project path so it round-trips cleanly.
                rel = os.path.relpath(best, os.path.dirname(__file__))
                out.append({"name": name, "path": rel})
    # Always include whatever the config currently points at, even if it
    # doesn't live under runs/detect (e.g. a manual pretrained file).
    cur = config.YOLO_MODEL_PATH
    if cur and not any(o["path"] == cur for o in out):
        out.insert(0, {"name": f"(custom) {os.path.basename(cur)}", "path": cur})
    return out


@app.route("/api/yolo_config", methods=["GET"])
def api_yolo_config():
    """Return current YOLO + replay tuning config for UI display."""
    return jsonify({
        "confidence": config.YOLO_CONFIDENCE,
        "iou_threshold": config.YOLO_IOU_THRESHOLD,
        "model_path": config.YOLO_MODEL_PATH,
        "available_models": _discover_yolo_models(),
        "tta": bool(getattr(config, "YOLO_TTA", False)),
        "conf_far": float(getattr(config, "YOLO_CONF_FAR", 0.05)),
        "conf_near": float(getattr(config, "YOLO_CONF_NEAR", 0.12)),
        "max_imgsz": int(getattr(config, "YOLO_MAX_IMGSZ", 1280)),
        "replay_sim_compression": bool(
            getattr(config, "REPLAY_SIMULATE_LIVE_COMPRESSION", False)),
        "replay_sim_quality": int(getattr(config, "REPLAY_SIM_JPEG_QUALITY", 60)),
        "iou_threshold": float(getattr(config, "YOLO_IOU_THRESHOLD", 0.85)),
        "tracker_config": str(getattr(config, "YOLO_TRACKER_CONFIG", "bytetrack_ftc.yaml")),
        "tripwire_min_track_age": int(
            getattr(config, "TRIPWIRE_MIN_TRACK_AGE_FRAMES", 5)),
    })


@app.route("/api/yolo_config", methods=["POST"])
def api_set_yolo_config():
    """Update YOLO + replay tuning config live."""
    data = request.get_json(silent=True) or {}

    if "confidence" in data:
        config.YOLO_CONFIDENCE = float(data["confidence"])
        if hasattr(detector, 'confidence'):
            detector.confidence = config.YOLO_CONFIDENCE
    if "iou_threshold" in data:
        config.YOLO_IOU_THRESHOLD = float(data["iou_threshold"])
        if hasattr(detector, 'iou_threshold'):
            detector.iou_threshold = config.YOLO_IOU_THRESHOLD
    if "tta" in data:
        config.YOLO_TTA = bool(data["tta"])
    if "conf_far" in data:
        config.YOLO_CONF_FAR = float(data["conf_far"])
    if "conf_near" in data:
        config.YOLO_CONF_NEAR = float(data["conf_near"])
    if "max_imgsz" in data:
        v = int(data["max_imgsz"])
        if v in (640, 960, 1280, 1600):
            config.YOLO_MAX_IMGSZ = v
    if "replay_sim_compression" in data:
        config.REPLAY_SIMULATE_LIVE_COMPRESSION = bool(data["replay_sim_compression"])
    if "replay_sim_quality" in data:
        config.REPLAY_SIM_JPEG_QUALITY = int(data["replay_sim_quality"])
    if "tripwire_min_track_age" in data:
        v = max(1, min(60, int(data["tripwire_min_track_age"])))
        config.TRIPWIRE_MIN_TRACK_AGE_FRAMES = v
        for t in (ramp_tracker_red, ramp_tracker_blue):
            if t is not None and hasattr(t, "set_min_track_age"):
                t.set_min_track_age(v)
    if "model_path" in data:
        new_path = data["model_path"]
        # Resolve relative paths against the project root so the dropdown
        # values (which are relative) load correctly.
        resolved = new_path
        if not os.path.isabs(new_path):
            resolved = os.path.join(os.path.dirname(__file__), new_path)
        if not os.path.isfile(resolved):
            return jsonify({"status": "error",
                            "message": f"Model file not found: {resolved}"}), 400
        try:
            detector.swap_model(resolved)
            config.YOLO_MODEL_PATH = new_path  # store the form the UI sent
            print(f"[MODEL] Hot-swapped to {new_path}")
        except Exception as e:
            return jsonify({"status": "error",
                            "message": f"swap_model failed: {e}"}), 500

    return jsonify({"status": "ok"})


@app.route("/api/roi", methods=["GET"])
def api_get_roi():
    """Return saved ROI and gate zone coordinates for both alliances."""
    data = load_roi_config()
    return jsonify(data)


@app.route("/api/roi", methods=["POST"])
def api_set_roi():
    """Save ROI and gate zone coordinates per alliance. Updates ramp trackers.

    Accepts:
        {
          "red":  {"roi": [...], "gate": [...], "exit": [...], "divider": [[x,y],[x,y]]},
          "blue": {"roi": [...], "gate": [...], "exit": [...], "divider": [[x,y],[x,y]]}
        }
    All fields are optional; only provided fields are updated.
    Polygons need 3+ points; divider is exactly 2 points (a line).
    """
    data = request.get_json(silent=True) or {}

    for alliance in ("red", "blue"):
        a_data = data.get(alliance)
        if a_data is None:
            continue
        for key in ("roi", "gate", "exit"):
            coords = a_data.get(key)
            if coords is not None:
                if not isinstance(coords, list) or len(coords) < 3:
                    return jsonify({"status": "error",
                                    "message": f"{alliance}.{key} must be a polygon with 3+ points"}), 400
        div = a_data.get("divider")
        if div is not None:
            if not isinstance(div, list) or len(div) != 2:
                return jsonify({"status": "error",
                                "message": f"{alliance}.divider must be [[x,y],[x,y]]"}), 400

    save_data = load_roi_config()
    # Drop the legacy global divider if it's still on disk from old configs.
    save_data.pop("divider", None)

    for alliance, tracker in [("red", ramp_tracker_red), ("blue", ramp_tracker_blue)]:
        a_data = data.get(alliance)
        if a_data is None:
            continue
        if alliance not in save_data:
            save_data[alliance] = {"roi": None, "gate_trip": None, "overflow_trip": None}
        if "roi" in a_data:
            save_data[alliance]["roi"] = a_data["roi"]
            if tracker is not None and a_data["roi"] is not None:
                tracker.set_roi(a_data["roi"])
        if "gate_trip" in a_data:
            save_data[alliance]["gate_trip"] = a_data["gate_trip"]
            if tracker is not None and a_data["gate_trip"] is not None:
                tracker.set_gate_trip(a_data["gate_trip"])
        if "overflow_trip" in a_data:
            save_data[alliance]["overflow_trip"] = a_data["overflow_trip"]
            if tracker is not None and a_data["overflow_trip"] is not None:
                tracker.set_overflow_trip(a_data["overflow_trip"])
        # Drop legacy fields if the client sends them (no-op storage).
        for legacy in ("gate", "exit", "divider"):
            if legacy in a_data:
                save_data[alliance].pop(legacy, None)

    save_roi_config(save_data)
    return jsonify({"status": "ok"})


_TRACKER_TUNABLE_FIELDS = (
    "overflow_start_fraction",
    "pass_match_radius_norm",
    "pass_gap_frames",
    "pass_min_frames",
    "post_commit_lockout_frames",
    "smoothing_window",
    "classified_max",
)


_setup_framesize = None  # stash the low-res framesize when boosting


@app.route("/api/camera/boost", methods=["POST"])
def api_camera_boost():
    """Switch the ESP32 to high resolution for match play.

    POST {"framesize": 16}   — push FHD (or whatever value) to the camera.
    POST {"reset": true}     — drop back to the setup-mode framesize.
    """
    global _setup_framesize
    data = request.get_json(silent=True) or {}

    if data.get("reset"):
        if _setup_framesize is not None:
            config.ESP32_DEFAULT_FRAMESIZE = _setup_framesize
            _setup_framesize = None
        configure_esp32_camera(force=True)
        return jsonify({"status": "ok", "mode": "setup",
                        "framesize": config.ESP32_DEFAULT_FRAMESIZE})

    fs = data.get("framesize")
    if fs is None:
        return jsonify({"status": "error",
                        "message": "Provide 'framesize' (int) or 'reset': true"}), 400
    fs = int(fs)
    if _setup_framesize is None:
        _setup_framesize = config.ESP32_DEFAULT_FRAMESIZE
    config.ESP32_DEFAULT_FRAMESIZE = fs
    configure_esp32_camera(force=True)
    return jsonify({"status": "ok", "mode": "boosted",
                    "framesize": fs,
                    "setup_framesize": _setup_framesize})



_score_events = []
_SCORE_EVENT_MAX = 200
_score_event_seq = 0
_score_event_lock = threading.Lock()


def _log_score_deltas(stream_id, prev, cur, include_pattern=False):
    """Compare two scorer snapshots and log any score change as an event.

    During the match we ONLY emit classified and overflow events. Pattern
    points are evaluated and emitted explicitly when the user ends AUTO
    or TELEOP (see api_advance_phase). This keeps the live log focused on
    what's happening per ball, with pattern bonus appearing as one
    summary event per period.
    """
    global _score_event_seq
    alliance = "red" if "red" in stream_id else ("blue" if "blue" in stream_id else stream_id)
    fields = [
        ("classified_count", "classified", config.POINTS_CLASSIFIED_TELEOP),
        ("overflow_count",   "overflow",   config.POINTS_OVERFLOW_TELEOP),
    ]
    if include_pattern:
        fields.append(("pattern_match_count", "pattern", config.POINTS_PATTERN_MATCH))
    diffs = []
    for key, label, pts in fields:
        d = cur.get(key, 0) - prev.get(key, 0)
        if d > 0:
            diffs.append({"label": label, "count": d, "points": d * pts})
    if not diffs:
        return
    with _score_event_lock:
        for diff in diffs:
            _score_event_seq += 1
            _score_events.append({
                "id": _score_event_seq,
                "t": time.time(),
                "alliance": alliance,
                **diff,
            })
        if len(_score_events) > _SCORE_EVENT_MAX:
            del _score_events[: len(_score_events) - _SCORE_EVENT_MAX]


PATTERN_SAMPLE_FRAMES = 45  # ~3s at 15fps


def _snapshot_and_lock_pattern(phase_name):
    """At end of AUTO/TELEOP, take ~3 seconds of recent ROI snapshots,
    sort each frame's balls from opposite-gate-end → gate-end (the
    gate_trip polygon centroid is the gate anchor), and lock the
    consensus (mode color per slot index) as the MOTIF pattern.

    Sampling across a window beats single-frame snapshots because:
      - YOLO has occasional per-frame drops; a 45-frame mode survives them
      - balls may briefly shift during the period-end pause
    """
    from collections import Counter
    for alliance, scorer, tracker in (
            ("red", scorer_red, ramp_tracker_red),
            ("blue", scorer_blue, ramp_tracker_blue)):
        if scorer is None or tracker is None:
            continue
        history = tracker.get_recent_balls_history(PATTERN_SAMPLE_FRAMES) \
            if hasattr(tracker, "get_recent_balls_history") else []
        if not history:
            scorer.lock_pattern([])
            print(f"[PATTERN] {alliance} ({phase_name}): no history → locked empty")
            continue
        roi = tracker.roi
        gate_anchor = getattr(tracker, "gate_trip_poly", None)

        # Per-slot color counters across the sampling window.
        slot_counts = [Counter() for _ in range(9)]
        for _frame_idx, balls in history:
            ordered = _sort_balls_for_pattern(balls, roi, gate_anchor)
            for i, color in enumerate(ordered[:9]):
                slot_counts[i][color] += 1

        # Mode per slot. A slot with no observations is dropped (treat
        # as "no ball there" — the pattern just becomes shorter).
        consensus = []
        for i in range(9):
            if not slot_counts[i]:
                break  # no balls observed at this slot index → end of ramp
            top_color, _votes = slot_counts[i].most_common(1)[0]
            consensus.append(top_color)
        scorer.lock_pattern(consensus)
        print(f"[PATTERN] {alliance} ({phase_name}): "
              f"{len(history)} frames sampled → locked {consensus}")


def _sort_balls_for_pattern(balls, roi_poly, gate_anchor):
    """Sort detected balls along the OPPOSITE-of-gate → gate axis.
    Index 0 = far end (opposite gate), index N-1 = at the gate.

    Balls are expected to carry normalized coords ('x_norm', 'y_norm')
    as written into the tripwire history. The polygon coords are also
    normalized, so all the math stays in one coordinate system.

    Sort key: distance from the gate-trip centroid, DESCENDING.
    """
    candidates = [b for b in balls if b.get("color") in ("G", "P")]
    if not candidates:
        return []
    if gate_anchor and len(gate_anchor) >= 3:
        gx = sum(p[0] for p in gate_anchor) / len(gate_anchor)
        gy = sum(p[1] for p in gate_anchor) / len(gate_anchor)
        def dist2(b):
            dx = b.get("x_norm", 0) - gx
            dy = b.get("y_norm", 0) - gy
            return dx * dx + dy * dy
        # Negate so ascending sort puts FAR (large distance) first.
        ordered = sorted(candidates, key=lambda b: -dist2(b))
    else:
        # Fallback: left-to-right by x_norm.
        ordered = sorted(candidates, key=lambda b: b.get("x_norm", 0))
    return [b["color"] for b in ordered]


def _emit_phase_pattern_event(phase_name):
    """Emit one pattern score event per alliance for the just-ended phase."""
    global _score_event_seq
    for alliance, scorer in (("red", scorer_red), ("blue", scorer_blue)):
        if scorer is None:
            continue
        snap = scorer.get_scores()
        match_count = snap.get("pattern_match_count", 0)
        if match_count == 0:
            continue
        with _score_event_lock:
            _score_event_seq += 1
            _score_events.append({
                "id": _score_event_seq,
                "t": time.time(),
                "alliance": alliance,
                "label": f"pattern ({phase_name})",
                "count": match_count,
                "points": match_count * config.POINTS_PATTERN_MATCH,
            })
            if len(_score_events) > _SCORE_EVENT_MAX:
                del _score_events[: len(_score_events) - _SCORE_EVENT_MAX]


@app.route("/api/score_events", methods=["GET"])
def api_score_events():
    """Return score events newer than the given id (default: all)."""
    since = int(request.args.get("since", 0))
    with _score_event_lock:
        out = [e for e in _score_events if e["id"] > since]
    return jsonify({"events": out})


@app.route("/api/tripwire_debug", methods=["GET"])
def api_tripwire_debug():
    """Live debug view of the tripwire counters: active tracks +
    chronological event log for each tripwire (gate + overflow), per
    alliance. Used by the dashboard's TRIPWIRE TUNING panel.

    Query params (optional):
      since_red_gate, since_red_overflow, since_blue_gate, since_blue_overflow
      — last seq id seen by the client, so only newer events are returned.
    """
    def _state_for(tr, prefix):
        if tr is None:
            return None
        since = {
            "gate":     int(request.args.get(f"since_{prefix}_gate", 0)),
            "overflow": int(request.args.get(f"since_{prefix}_overflow", 0)),
        }
        return tr.get_debug_state(since)
    def _csrt_state(t):
        if t is None or not hasattr(t, "get_debug_state"):
            return None
        return t.get_debug_state()
    def _csrt_events(t, since):
        if t is None or not hasattr(t, "get_events_since"):
            return []
        return t.get_events_since(since)
    since_csrt_red = int(request.args.get("since_csrt_red", 0))
    since_csrt_blue = int(request.args.get("since_csrt_blue", 0))
    return jsonify({
        "red":  _state_for(ramp_tracker_red, "red"),
        "blue": _state_for(ramp_tracker_blue, "blue"),
        "csrt_red":  _csrt_state(csrt_tracker_red),
        "csrt_blue": _csrt_state(csrt_tracker_blue),
        "csrt_red_events":  _csrt_events(csrt_tracker_red,  since_csrt_red),
        "csrt_blue_events": _csrt_events(csrt_tracker_blue, since_csrt_blue),
        "csrt_cap": int(getattr(config, "CSRT_MAX_ACTIVE_TRACKS", 12)),
    })


@app.route("/api/exit_events", methods=["GET"])
def api_exit_events():
    """Return recent exit events from both trackers, with alliance tag."""
    since = int(request.args.get("since", 0))
    out = []
    for alliance, tr in [("red", ramp_tracker_red), ("blue", ramp_tracker_blue)]:
        if tr is None or not hasattr(tr, "get_exit_events"):
            continue
        for ev in tr.get_exit_events(since_frame=since):
            out.append({
                "alliance": alliance,
                "color": ev["color"],
                "t": ev["t"],
                "frame": ev["frame"],
            })
    out.sort(key=lambda e: e["t"])
    return jsonify({"events": out})


@app.route("/api/camera/configure", methods=["POST"])
def api_camera_configure():
    """Change ESP32 camera settings on the fly.

    POST {"framesize": N}        — change resolution
    POST {"quality": N}          — change JPEG quality
    POST {"framesize": N, "quality": N}  — both at once
    POST {"vflip": 0|1}          — vertical flip
    POST {"hmirror": 0|1}        — horizontal mirror

    Framesize values (Freenove ESP32-S3):
      8=VGA(640x480), 9=SVGA(800x600), 10=XGA(1024x768),
      11=HD(1280x720), 12=SXGA(1280x1024), 13=UXGA(1600x1200)
    """
    data = request.get_json(silent=True) or {}
    results = {}
    for var in ("framesize", "quality", "vflip", "hmirror"):
        if var not in data:
            continue
        val = int(data[var])
        try:
            url = f"{config.ESP32_CONTROL_URL}?var={var}&val={val}"
            resp = http_requests.get(url, timeout=3)
            results[var] = {"val": val, "ok": resp.status_code == 200}
            if var == "framesize":
                config.ESP32_DEFAULT_FRAMESIZE = val
            elif var == "quality":
                config.ESP32_DEFAULT_QUALITY = val
        except Exception as e:
            results[var] = {"val": val, "ok": False, "error": str(e)}
    return jsonify({"status": "ok", "results": results})


@app.route("/api/camera/reapply", methods=["POST"])
def api_camera_reapply():
    """Re-push framesize/quality to the ESP32 without waiting for a reconnect.

    Useful if you notice image quality has drifted mid-match (e.g. a
    soft ESP32 reset that preserved the Wi-Fi socket but reset camera
    sensor settings) — hitting this endpoint re-asserts our config.
    """
    try:
        configure_esp32_camera(force=True)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/camera/stats", methods=["GET"])
def api_camera_stats():
    """Return live camera health: resolution + label, grab/process fps,
    cumulative frames and dropped, drop ratio."""
    with camera_stats_lock:
        snap = dict(camera_stats)
    total = snap.get("frames_total", 0)
    dropped = snap.get("dropped_total", 0)
    snap["drop_ratio"] = (dropped / total) if total > 0 else 0.0
    return jsonify(snap)


@app.route("/api/tracker/params", methods=["GET"])
def api_get_tracker_params():
    """Return current tunable tracker parameters (read from the red tracker;
    params are kept identical on both alliances)."""
    if ramp_tracker_red is None:
        return jsonify({"status": "error", "message": "Tracker not initialized"}), 503
    out = {}
    for key in _TRACKER_TUNABLE_FIELDS:
        if hasattr(ramp_tracker_red, key):
            out[key] = getattr(ramp_tracker_red, key)
    return jsonify(out)


@app.route("/api/tracker/params", methods=["POST"])
def api_set_tracker_params():
    """Update one or more tunable tracker parameters on BOTH alliances.

    Request body: JSON object with any subset of _TRACKER_TUNABLE_FIELDS.
    Unknown fields are ignored. Int fields are coerced from floats.
    """
    data = request.get_json(silent=True) or {}
    applied = {}
    int_fields = {"pass_gap_frames", "pass_min_frames",
                  "post_commit_lockout_frames",
                  "smoothing_window", "classified_max"}
    for key in _TRACKER_TUNABLE_FIELDS:
        if key not in data:
            continue
        try:
            val = data[key]
            val = int(val) if key in int_fields else float(val)
        except (TypeError, ValueError):
            return jsonify({"status": "error",
                            "message": f"{key} must be numeric"}), 400
        for tracker in (ramp_tracker_red, ramp_tracker_blue):
            if tracker is not None and hasattr(tracker, key):
                setattr(tracker, key, val)
        applied[key] = val
    return jsonify({"status": "ok", "applied": applied})


@app.route("/api/ramp", methods=["GET"])
def api_get_ramp():
    """Return current ramp tracker state for both alliances."""
    def _state(tracker):
        if tracker is None:
            return {"sequence": [], "count": 0,
                    "classified_total": 0, "overflow_total": 0,
                    "exited_total": 0, "occupancy": 0}
        seq = tracker.get_sequence()
        totals = tracker.get_totals()
        return {
            "sequence": seq,
            "count": len(seq),
            "classified_total": totals["classified"],
            "overflow_total": totals["overflow"],
            "exited_total": totals["exited"],
            "occupancy": totals["occupancy"],
        }
    return jsonify({"red": _state(ramp_tracker_red),
                    "blue": _state(ramp_tracker_blue)})


@app.route("/api/ramp/reset", methods=["POST"])
def api_reset_ramp():
    """Reset both ramp trackers and scorers for a new match."""
    if ramp_tracker_red is not None:
        ramp_tracker_red.reset()
    if ramp_tracker_blue is not None:
        ramp_tracker_blue.reset()
    if csrt_tracker_red is not None:
        csrt_tracker_red.reset()
    if csrt_tracker_blue is not None:
        csrt_tracker_blue.reset()
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
    scorer_red.unlock_pattern()
    scorer_blue.unlock_pattern()
    scorer_red.update([])
    scorer_blue.update([])
    return jsonify({"status": "ok"})


@app.route("/api/matches", methods=["GET"])
def api_list_matches():
    """List available match footage files."""
    video_exts = (".mp4", ".mkv", ".avi", ".mov", ".webm")
    matches = []
    if os.path.isdir(MATCH_FOOTAGE_DIR):
        for f in sorted(os.listdir(MATCH_FOOTAGE_DIR)):
            if f.lower().endswith(video_exts):
                path = os.path.join(MATCH_FOOTAGE_DIR, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                matches.append({"name": f, "size_mb": round(size_mb, 1)})
    return jsonify({
        "matches": matches,
        "current": os.path.basename(_replay_current_video) if _replay_current_video else None,
    })


@app.route("/api/matches/play", methods=["POST"])
def api_play_match():
    """Switch replay to a different match video. Also resets the ramp tracker."""
    global _replay_switch_to, _replay_paused
    _replay_paused = True  # Pause when switching so user can adjust zones
    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    if not name:
        return jsonify({"status": "error", "message": "Missing name"}), 400

    path = os.path.join(MATCH_FOOTAGE_DIR, name)
    if not os.path.isfile(path):
        return jsonify({"status": "error", "message": f"Not found: {name}"}), 404

    # Reset trackers for new match
    if ramp_tracker_red is not None:
        ramp_tracker_red.reset()
    if ramp_tracker_blue is not None:
        ramp_tracker_blue.reset()
    if csrt_tracker_red is not None:
        csrt_tracker_red.reset()
    if csrt_tracker_blue is not None:
        csrt_tracker_blue.reset()
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
    scorer_red.unlock_pattern()
    scorer_blue.unlock_pattern()
    scorer_red.update([])
    scorer_blue.update([])

    _replay_switch_to = path
    return jsonify({"status": "ok", "playing": name})


@app.route("/api/label/state", methods=["GET"])
def api_label_state():
    """Current label-mode state — counts per cell + recent events."""
    with _label_lock:
        return jsonify({
            "label_mode": _label_mode,
            "alliance": _label_alliance,
            "video": _label_video_basename,
            "counts": _label_counts,
            "events": list(_label_events),
            "path": _label_path() if _label_mode else None,
        })


@app.route("/api/label/increment", methods=["POST"])
def api_label_increment():
    """Increment (or decrement) a manual count cell. Records frame +
    timestamp on every +1 click; on −1, pops the most-recent matching
    event so undo removes the right entry from the timeline."""
    global _label_seq
    if not _label_mode:
        return jsonify({"ok": False, "reason": "not in label mode"}), 400
    data = request.get_json() or {}
    alliance = data.get("alliance")
    line = data.get("line")
    delta = int(data.get("delta", 1))
    if alliance not in ("red", "blue") or line not in ("classified", "overflow"):
        return jsonify({"ok": False, "reason": "bad alliance/line"}), 400
    # Reject writes to the alliance not active in THIS labeling pass.
    if _label_alliance != "both" and alliance != _label_alliance:
        return jsonify({
            "ok": False,
            "reason": f"this pass is locked to {_label_alliance}; "
                      f"restart with --alliance {alliance} to label that side",
        }), 403
    with _label_lock:
        new_total = max(0, _label_counts[alliance][line] + delta)
        _label_counts[alliance][line] = new_total
        if delta > 0:
            _label_seq += 1
            _label_events.append({
                "seq": _label_seq,
                "frame": int(_replay_current_frame),
                "t": time.time(),
                "alliance": alliance,
                "line": line,
            })
        elif delta < 0:
            for i in range(len(_label_events) - 1, -1, -1):
                ev = _label_events[i]
                if ev["alliance"] == alliance and ev["line"] == line:
                    _label_events.pop(i)
                    break
        _label_save()
        return jsonify({"ok": True, "counts": _label_counts})


@app.route("/api/label/reset", methods=["POST"])
def api_label_reset():
    """Reset only the alliance currently being labeled. Pass --alliance
    both to reset everything in one shot."""
    global _label_seq
    if not _label_mode:
        return jsonify({"ok": False, "reason": "not in label mode"}), 400
    targets = ("red", "blue") if _label_alliance == "both" else (_label_alliance,)
    with _label_lock:
        for a in targets:
            for ln in ("classified", "overflow"):
                _label_counts[a][ln] = 0
        # Drop only events for the alliances we cleared.
        _label_events[:] = [e for e in _label_events
                            if e["alliance"] not in targets]
        _label_save()
    return jsonify({"ok": True, "reset": list(targets)})


@app.route("/api/benchmark", methods=["GET"])
def api_benchmark():
    """Compare the live auto-counter against pre-recorded ground truth.

    Auto events come from the alliance tripwire counters' internal
    _events buffer (one per actual count fire). Ground-truth events
    come from labels/<clip>.json. Matching is greedy by frame distance:
    each GT event finds the closest unused auto event in the same
    (alliance, line) within ±tolerance frames.

    Returns per-cell counts (gt vs auto), match counts (TP/FP/FN), and
    per-event detail for debugging which GT events were missed and
    which auto events fired without a GT counterpart."""
    if _ground_truth is None:
        return jsonify({"available": False,
                        "reason": "no labels file loaded"})

    # Default 150 = ~5 sec at 30fps. Ground-truth timing varies because
    # human reaction time + balls moving at different rates between
    # entry and tripwire — a tight window over-reports FP/FN. Slider in
    # the UI lets you tune live.
    tol = int(request.args.get("tol", 150))

    # If a clip start_frame is set (camera unstable at clip start),
    # filter both auto AND ground-truth events to frames >= start_frame.
    # Otherwise the GT events recorded in the pre-gameplay region would
    # all show up as missed FNs and inflate the error rate.
    clip_start = 0
    if ramp_tracker_red is not None and hasattr(ramp_tracker_red, "start_frame"):
        clip_start = int(ramp_tracker_red.start_frame or 0)

    # Collect auto events from both alliance tripwire counters.
    auto_events = []
    for alliance, tracker in [("red", ramp_tracker_red),
                              ("blue", ramp_tracker_blue)]:
        if tracker is None:
            continue
        for line_attr, line_name in [("gate_trip", "classified"),
                                     ("overflow_trip", "overflow")]:
            tw = getattr(tracker, line_attr, None)
            if tw is None:
                continue
            for ev in getattr(tw, "_events", []):
                auto_events.append({
                    "alliance": alliance,
                    "line": line_name,
                    "frame": int(ev.get("frame", 0)),
                    "tid": ev.get("tid"),
                    "color": ev.get("color"),
                })

    gt_events = list(_ground_truth.get("events", []))
    # Apply start_frame filter symmetrically: drop pre-gameplay GT events
    # AND any auto events that somehow slipped through (shouldn't happen
    # given tripwire_counter is gated, but defense-in-depth).
    if clip_start > 0:
        gt_events = [e for e in gt_events
                     if int(e.get("frame", 0)) >= clip_start]
        auto_events = [e for e in auto_events
                       if int(e.get("frame", 0)) >= clip_start]

    # Greedy match. For each GT event, find the closest unused auto
    # event of the same (alliance, line) within ±tol frames.
    auto_used = [False] * len(auto_events)
    matched = []     # (gt, auto, dframes)
    missed = []      # GT had it but no auto fired in window
    for gt in gt_events:
        best = None  # (delta, auto_idx)
        for i, au in enumerate(auto_events):
            if auto_used[i]:
                continue
            if au["alliance"] != gt["alliance"]:
                continue
            if au["line"] != gt["line"]:
                continue
            df = abs(au["frame"] - gt["frame"])
            if df > tol:
                continue
            if best is None or df < best[0]:
                best = (df, i)
        if best is None:
            missed.append(gt)
        else:
            auto_used[best[1]] = True
            matched.append({
                "gt_frame": gt["frame"],
                "auto_frame": auto_events[best[1]]["frame"],
                "delta": best[0],
                "alliance": gt["alliance"],
                "line": gt["line"],
            })
    false_positives = [a for i, a in enumerate(auto_events) if not auto_used[i]]

    # Per-cell rollup
    def _cell(alliance, line):
        gt_n = sum(1 for e in gt_events
                   if e["alliance"] == alliance and e["line"] == line)
        au_n = sum(1 for e in auto_events
                   if e["alliance"] == alliance and e["line"] == line)
        tp = sum(1 for m in matched
                 if m["alliance"] == alliance and m["line"] == line)
        fn = gt_n - tp
        fp = au_n - tp
        # Symmetric raw-count accuracy (min/max).
        raw_acc = (min(gt_n, au_n) / max(gt_n, au_n)) if max(gt_n, au_n) else None
        # F1 over event-aligned matches.
        if (tp + fp) and (tp + fn):
            prec = tp / (tp + fp); rec = tp / (tp + fn)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0
        elif gt_n == 0 and au_n == 0:
            f1 = None
        else:
            f1 = 0.0
        return {"gt": gt_n, "auto": au_n, "tp": tp, "fp": fp, "fn": fn,
                "raw_acc": raw_acc, "f1": f1}

    cells = {a: {l: _cell(a, l) for l in ("classified", "overflow")}
             for a in ("red", "blue")}

    # Overall raw + F1
    total_gt = len(gt_events); total_auto = len(auto_events)
    total_tp = len(matched)
    overall_raw = (min(total_gt, total_auto) / max(total_gt, total_auto)) \
                  if max(total_gt, total_auto) else None
    payload = {
        "available": True,
        "video": _ground_truth.get("video"),
        "tolerance_frames": tol,
        "start_frame": clip_start,
        "totals": {
            "gt": total_gt, "auto": total_auto,
            "tp": total_tp, "fp": len(false_positives),
            "fn": len(missed),
            "raw_acc": overall_raw,
        },
        "cells": cells,
        "missed_tail": missed[-20:],         # GT events with no auto match
        "false_pos_tail": false_positives[-20:],  # auto events with no GT
        "matched_tail": matched[-20:],
    }
    # Persist to disk so an abrupt server stop doesn't lose the run.
    # Includes the FULL miss + FP lists (not just the tail) for forensics.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(here, "results")
        os.makedirs(rdir, exist_ok=True)
        rpath = os.path.join(rdir, _ground_truth["video"] + ".json")
        on_disk = dict(payload)
        on_disk["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        on_disk["source"] = "live_run"
        on_disk["missed_full"] = missed
        on_disk["false_pos_full"] = false_positives
        on_disk["matched_full"] = matched
        # Atomic write so concurrent reads never see a partial file.
        tmp = rpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(on_disk, f, indent=2)
        os.replace(tmp, rpath)
    except (OSError, KeyError):
        pass
    return jsonify(payload)


@app.route("/api/version", methods=["GET"])
def api_version():
    """Returns mtime of csrt_tracker.py + app.py + index.html so the
    dashboard can show whether browser/server are running fresh code."""
    here = os.path.dirname(os.path.abspath(__file__))
    files = {
        "csrt_tracker.py": os.path.join(here, "csrt_tracker.py"),
        "app.py":          os.path.join(here, "app.py"),
        "templates/index.html": os.path.join(here, "templates", "index.html"),
    }
    out = {}
    import datetime as _dt
    for label, path in files.items():
        if os.path.exists(path):
            mt = os.path.getmtime(path)
            out[label] = _dt.datetime.fromtimestamp(mt).strftime(
                "%Y-%m-%d %H:%M:%S")
    # Marker we know was added in the dedup commit so we can confirm
    # the server is running the NEW csrt_tracker code.
    out["server_has_dedup"] = hasattr(
        __import__("csrt_tracker").MultiBallTracker(), "dedup_radius_px")
    return jsonify(out)


@app.route("/api/save_snapshot", methods=["POST"])
def api_save_snapshot():
    """Save the JSON snapshot the dashboard built to disk under
    ./snapshots/<timestamp>.txt. Survives browser clipboard failures
    and gives a permanent record we can grep later."""
    import time as _t
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    if not text:
        return jsonify({"ok": False, "reason": "empty"}), 400
    snap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    ts = _t.strftime("%Y%m%d-%H%M%S", _t.localtime())
    fname = f"snap-{ts}.txt"
    fpath = os.path.join(snap_dir, fname)
    with open(fpath, "w") as f:
        f.write(text)
    print(f"[snap] wrote {fpath} ({len(text)} bytes)", flush=True)
    return jsonify({"ok": True, "path": fpath, "filename": fname})


@app.route("/api/replay/pause", methods=["POST"])
def api_replay_pause():
    """Pause or resume video replay."""
    global _replay_paused
    data = request.get_json(silent=True) or {}
    if "paused" in data:
        _replay_paused = bool(data["paused"])
    else:
        _replay_paused = not _replay_paused  # Toggle
    return jsonify({"status": "ok", "paused": _replay_paused})


@app.route("/api/replay/status", methods=["GET"])
def api_replay_status():
    """Get replay state including seek position + playback speed."""
    total = _replay_total_frames
    cur = _replay_current_frame
    fps = _replay_src_fps or 30
    return jsonify({
        "paused": _replay_paused,
        "current": os.path.basename(_replay_current_video) if _replay_current_video else None,
        "current_frame": cur,
        "total_frames": total,
        "current_sec": round(cur / fps, 1) if fps else 0,
        "total_sec": round(total / fps, 1) if fps else 0,
        "src_fps": round(fps, 1),
        "playback_fps": round(_replay_target_fps, 1),
    })


@app.route("/api/replay/speed", methods=["POST"])
def api_replay_speed():
    """Adjust replay playback FPS. Body either {"fps": <int>} for an
    absolute target or {"mult": <float>} to multiply current FPS."""
    global _replay_target_fps
    data = request.get_json(silent=True) or {}
    if "fps" in data:
        _replay_target_fps = max(1.0, min(120.0, float(data["fps"])))
    elif "mult" in data:
        _replay_target_fps = max(1.0, min(120.0,
                                          _replay_target_fps * float(data["mult"])))
    return jsonify({"status": "ok", "playback_fps": round(_replay_target_fps, 1)})


@app.route("/api/replay/seek", methods=["POST"])
def api_replay_seek():
    """Seek to a specific frame or time in the current video."""
    global _replay_seek_to
    data = request.get_json(silent=True) or {}
    if "frame" in data:
        _replay_seek_to = int(data["frame"])
    elif "seconds" in data:
        _replay_seek_to = int(float(data["seconds"]) * _replay_src_fps)
    else:
        return jsonify({"status": "error", "message": "Provide 'frame' or 'seconds'"}), 400
    return jsonify({"status": "ok", "seek_to": _replay_seek_to})


# ============= Main =============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FTC DECODE Scoring Vision System")
    parser.add_argument("--port", type=int, default=8089,
                        help="Web server port (default: 8089)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--usb", nargs="?", const=0, type=int, default=None,
                        metavar="INDEX",
                        help="Use USB webcam instead of ESP32 WiFi (default index: 0)")
    parser.add_argument("--capture", action="store_true",
                        help="Use /capture polling instead of MJPEG stream (more compatible)")
    parser.add_argument("--stream-url", default=None, metavar="URL",
                        help="Override ESP32 stream URL (e.g. http://192.168.4.1/stream)")
    parser.add_argument("--framesize", type=int, default=None,
                        help="Override ESP32 framesize enum value (e.g. 14=SXGA 1280x1024, "
                             "15=UXGA 1600x1200, 16=FHD 1920x1080). Check [CAM] log "
                             "to confirm the camera accepted it.")
    parser.add_argument("--yolo-model", default=None, metavar="PATH",
                        help="Path to YOLO .pt model file")
    parser.add_argument("--detector", default="yolo", choices=["yolo"],
                        help="Detection backend (default: yolo)")
    parser.add_argument("--tracker", default="simple", choices=["simple", "full"],
                        help="Ramp tracker: 'simple' (default; temporally-smoothed "
                             "live count, no cumulative state) or 'full' (legacy "
                             "RampTracker with classified/overflow bookkeeping)")
    parser.add_argument("--replay", default=None, metavar="VIDEO",
                        help="Replay a video file through the dashboard (downscaled to ESP32 quality)")
    parser.add_argument("--replay-fps", type=int, default=30,
                        help="Replay FPS (default: 15, similar to ESP32 stream)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Don't loop the replay video")
    parser.add_argument("--live", action="store_true",
                        help="Force live ESP32-S3 camera (clears --replay/--usb). "
                             "Same as default ESP32 path, but explicit.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug/setup mode. Streams raw camera feed with zone "
                             "overlays, NO YOLO detection. Use to draw zones, tune "
                             "resolution/quality, and verify camera setup before "
                             "running a real match.")
    parser.add_argument("--label", action="store_true",
                        help="Ground-truth labeling mode. NO YOLO/CSRT runs — just "
                             "the raw video feed + manual count buttons (Q/W/A/S "
                             "hotkeys). Counts + per-click timestamps save to "
                             "labels/<videoname>.json. Use to collect ground "
                             "truth that the auto-counter can later be benchmarked "
                             "against.")
    parser.add_argument("--label-speed", type=float, default=0.5,
                        help="Replay speed multiplier in label mode "
                             "(default: 0.5 = half-speed so eyes can keep up).")
    parser.add_argument("--auto-run", action="store_true",
                        help="Benchmark mode: auto-press START MATCH 1s "
                             "after the replay loads + auto-exit when the "
                             "clip ends. Combine with --no-loop. Use this "
                             "in a batch runner to process every labeled "
                             "clip without manual clicks.")
    parser.add_argument("--alliance", choices=["red", "blue", "both"],
                        default="both",
                        help="Lock label mode to ONE alliance per pass — "
                             "the UI hides the other side and hotkeys "
                             "collapse to two keys (Q=classified, W=overflow). "
                             "The JSON file holds both alliances; the second "
                             "pass with the other alliance flag adds to it. "
                             "'both' (default) = legacy 4-button mode.")
    parser.add_argument("--enhance-roi", action="store_true",
                        help="Apply cheap post-upscale enhancement (cubic + "
                             "unsharp mask + CLAHE) to each alliance ROI crop. "
                             "Toggleable live via /api/camera/enhance.")
    parser.add_argument("--dnn-sr", action="store_true",
                        help="Also run FSRCNN 2× super-resolution on small "
                             "ROI crops before upscaling. Implies --enhance-roi. "
                             "Needs opencv-contrib-python and "
                             "models/FSRCNN_x2.pb (auto-downloaded on first use).")
    args = parser.parse_args()

    if args.enhance_roi or args.dnn_sr:
        import sys as _sys
        _sys.modules[__name__]._enhance_roi_enabled = True
        print("[ENHANCE] ROI enhancement pipeline enabled "
              "(cubic interp + unsharp + CLAHE)")
    if args.dnn_sr:
        import sys as _sys
        if _try_init_dnn_sr():
            _sys.modules[__name__]._dnn_sr_enabled = True
            print("[ENHANCE] DNN super-resolution (FSRCNN 2×) enabled")
        else:
            print("[ENHANCE] DNN super-resolution unavailable, continuing without it")

    if args.debug:
        args.replay = None
        args.usb = None
        args.live = True
        import sys as _sys
        _sys.modules[__name__]._debug_mode = True
        print("[DEBUG] Setup/debug mode — NO YOLO detection, zone overlays only")

    if args.auto_run:
        if not args.replay:
            raise SystemExit("--auto-run requires --replay (it's for batch benchmarking)")
        _auto_run = True
        # Force single-pass replay so we get one clean run per clip.
        args.no_loop = True
        # Schedule the match-start: a tiny background thread that waits
        # for the grab loop to start producing frames, then POSTs the
        # equivalent of "press START MATCH" via direct global mutation.
        def _kick_off():
            # Wait until the grab loop has read the first frame, then
            # call the same logic that POST /api/match/start runs (so
            # tripwires are reset to fresh state at frame 0 of source).
            global _replay_paused
            time.sleep(2.0)
            if ramp_tracker_red is not None: ramp_tracker_red.reset()
            if ramp_tracker_blue is not None: ramp_tracker_blue.reset()
            if csrt_tracker_red is not None: csrt_tracker_red.reset()
            if csrt_tracker_blue is not None: csrt_tracker_blue.reset()
            if hasattr(detector, "reset_tracker"): detector.reset_tracker()
            scorer_red.unlock_pattern(); scorer_blue.unlock_pattern()
            scorer_red.update([]); scorer_blue.update([])
            with match_state_lock:
                match_state["phase"] = "AUTO"
                match_state["started_at"] = time.time()
                match_state["phase_duration"] = 0
                match_state["auto_snapshot"] = {"red": None, "blue": None}
                match_state["final_snapshot"] = {"red": None, "blue": None}
            _replay_paused = False
            print("[*] --auto-run: match started")
        threading.Thread(target=_kick_off, daemon=True).start()
        print("[*] --auto-run enabled — match auto-starts in 2s, "
              "exits at end of clip")

    if args.label:
        if not args.replay:
            raise SystemExit("--label requires --replay (you label against a clip)")
        _label_mode = True
        _label_alliance = args.alliance
        _label_video_basename = os.path.splitext(
            os.path.basename(args.replay))[0]
        _label_load()
        _replay_target_fps = max(1, int(round(30 * args.label_speed)))
        print(f"[LABEL] Ground-truth labeling mode — alliance: "
              f"{_label_alliance.upper()}")
        print(f"        Video: {args.replay}")
        print(f"        Saving to: labels/{_label_video_basename}.json")
        print(f"        Resumed: R cls={_label_counts['red']['classified']} "
              f"ovr={_label_counts['red']['overflow']} | "
              f"B cls={_label_counts['blue']['classified']} "
              f"ovr={_label_counts['blue']['overflow']}")
        print(f"        Speed: {args.label_speed:.2f}x ({_replay_target_fps} fps)")
        if _label_alliance == "both":
            print(f"        Hotkeys: Q=R-cls W=R-ovr A=B-cls S=B-ovr Z=undo")
        else:
            print(f"        Hotkeys: Q=classified W=overflow Z=undo "
                  f"(locked to {_label_alliance.upper()})")

    if args.live:
        args.replay = None
        args.usb = None
        print("[LIVE] ESP32-S3 camera mode requested — connect to the AP at "
              f"{config.ESP32_STREAM_URL.split('/stream')[0]} first.")

    use_usb = args.usb is not None

    # Override stream URL / framesize if provided
    if args.stream_url:
        config.ESP32_STREAM_URL = args.stream_url
    if args.framesize is not None:
        config.ESP32_DEFAULT_FRAMESIZE = args.framesize
        print(f"[CONFIG] Framesize override: {args.framesize}")

    # Initialize detector (skip entirely in debug mode — no YOLO needed)
    if args.debug:
        det_mode = "NONE (debug mode)"
    else:
        try:
            model_path = args.yolo_model or config.YOLO_MODEL_PATH
            from yolo_detector import YOLODetector
            # When using CSRT, disable ultralytics' built-in tracker
            # (model.track()) — we just want raw detections per frame
            # and run our own correlation-filter trackers downstream.
            backend = getattr(config, "TRACKER_BACKEND", "csrt").lower()
            tracking_enabled = (backend != "csrt")
            detector = YOLODetector(model_path=model_path,
                                     tracking_enabled=tracking_enabled)
            det_mode = f"YOLO ({model_path}) [{backend}]"
        except ImportError as e:
            print(f"[!] Detector init failed (missing dependency): {e}")
            print("    Install ultralytics: pip install ultralytics")
            import sys
            sys.exit(1)
        except Exception as e:
            print(f"[!] Detector init failed: {e}")
            print("    Check that the model file exists and ultralytics is installed")
            import sys
            sys.exit(1)

    # Initialize dual tripwire counters (one per alliance).
    # Plus per-alliance CSRT MultiBallTracker if TRACKER_BACKEND="csrt".
    try:
        from tripwire_counter import TripwireCounter as _Tracker
        tracker_mode = "TRIPWIRE (event-based gate + overflow counting)"
        _tk_kwargs = dict(
            memory_frames=int(getattr(config, "TRIPWIRE_TRACK_MEMORY_FRAMES", 600)),
            trail_length=int(getattr(config, "TRIPWIRE_TRAIL_LENGTH", 30)),
            min_track_age_frames=int(getattr(config, "TRIPWIRE_MIN_TRACK_AGE_FRAMES", 5)),
            settle_motion_threshold_px=float(
                getattr(config, "TRIPWIRE_SETTLE_MOTION_THRESHOLD_PX", 8.0)),
            settle_frames_required=int(
                getattr(config, "TRIPWIRE_SETTLE_FRAMES_REQUIRED", 5)),
            settled_match_radius_px=float(
                getattr(config, "TRIPWIRE_SETTLED_MATCH_RADIUS_PX", 30.0)),
            settled_eviction_frames=int(
                getattr(config, "TRIPWIRE_SETTLED_EVICTION_FRAMES", 150)),
            transit_eviction_frames=int(
                getattr(config, "TRIPWIRE_TRANSIT_EVICTION_FRAMES", 10)),
            settled_require_color=bool(
                getattr(config, "TRIPWIRE_SETTLED_REQUIRE_COLOR", True)),
        )
        ramp_tracker_red = _Tracker(**_tk_kwargs)
        ramp_tracker_blue = _Tracker(**_tk_kwargs)
        # Note: per-clip start_frame gate is applied later, after the
        # per-clip ROI block runs (it sets _roi_per_clip_path which we
        # need to know which clip's start_frame to load).
        # CSRT multi-ball trackers (per alliance). Bypassed when backend
        # is "bytetrack" — ultralytics' tracker assigns track_ids itself.
        # (No `global` needed — we're at module level here.)
        if getattr(config, "TRACKER_BACKEND", "csrt").lower() == "csrt":
            from csrt_tracker import MultiBallTracker
            _csrt_kwargs = dict(
                max_lost_frames=int(getattr(config, "CSRT_MAX_LOST_FRAMES", 30)),
                match_iou=float(getattr(config, "CSRT_MATCH_IOU", 0.20)),
                max_frames_without_yolo=int(
                    getattr(config, "CSRT_MAX_FRAMES_WITHOUT_YOLO", 60)),
                trail_length=int(getattr(config, "TRIPWIRE_TRAIL_LENGTH", 30)),
                ghost_max_frames=int(getattr(config, "CSRT_GHOST_MAX_FRAMES", 45)),
                ghost_match_radius_px=int(
                    getattr(config, "CSRT_GHOST_MATCH_RADIUS_PX", 40)),
                ghost_require_color=bool(
                    getattr(config, "CSRT_GHOST_REQUIRE_COLOR", True)),
                max_active_tracks=int(
                    getattr(config, "CSRT_MAX_ACTIVE_TRACKS", 12)),
                match_center_px=int(
                    getattr(config, "CSRT_MATCH_CENTER_PX", 35)),
                dedup_radius_px=int(
                    getattr(config, "CSRT_DEDUP_RADIUS_PX", 25)),
            )
            csrt_tracker_red = MultiBallTracker(**_csrt_kwargs)
            csrt_tracker_blue = MultiBallTracker(**_csrt_kwargs)
            print(f"  CSRT: per-ball correlation-filter trackers active")
        print(f"  Tracker: {tracker_mode}")
        # Load saved ROI/tripwire config for each alliance
        roi_data = load_roi_config()
        for alliance, tracker in [("red", ramp_tracker_red), ("blue", ramp_tracker_blue)]:
            a_data = roi_data.get(alliance, {})
            if a_data.get("roi"):
                tracker.set_roi(a_data["roi"])
                print(f"  Loaded {alliance} ROI ({len(a_data['roi'])} pts)")
            if a_data.get("gate_trip"):
                tracker.set_gate_trip(a_data["gate_trip"])
                print(f"  Loaded {alliance} Gate-Trip ({len(a_data['gate_trip'])} pts)")
            if a_data.get("overflow_trip"):
                tracker.set_overflow_trip(a_data["overflow_trip"])
                print(f"  Loaded {alliance} Overflow-Trip ({len(a_data['overflow_trip'])} pts)")
    except ImportError:
        print("[i] ramp_tracker module not found — using direct detection mode")
        ramp_tracker_red = None
        ramp_tracker_blue = None

    # Per-clip ROI: load/save zones from roi_configs/<clipname>.json so
    # each clip's camera angle has its own polygons. First-time runs
    # fall back to the global roi_config.json so you have something to
    # tweak; once you Save, it writes the per-clip file instead.
    if args.replay:
        clip_base = os.path.splitext(os.path.basename(args.replay))[0]
        _roi_per_clip_path = os.path.join(ROI_CONFIGS_DIR, clip_base + ".json")
        print(f"[ROI] Per-clip zones file: roi_configs/{clip_base}.json "
              f"({'exists' if os.path.exists(_roi_per_clip_path) else 'will be created on save'})")
        # Optional start_frame: tripwires won't fire any counts until
        # _replay_current_frame >= start_frame. Useful for clips whose
        # opening seconds have camera shifts / pre-match handling that
        # would otherwise be counted as false positives.
        clip_start = 0
        if os.path.exists(_roi_per_clip_path):
            try:
                with open(_roi_per_clip_path) as f:
                    clip_start = int(json.load(f).get("start_frame", 0))
            except (json.JSONDecodeError, IOError, ValueError, TypeError):
                pass
        if clip_start > 0:
            print(f"[ROI] start_frame for this clip: {clip_start} "
                  f"(skip counting until f{clip_start} = ~{clip_start/30.0:.1f}s)")
            # Apply the gate now that ramp trackers exist (they were
            # constructed earlier with start_frame=0 by default).
            if ramp_tracker_red is not None:
                ramp_tracker_red.start_frame = clip_start
            if ramp_tracker_blue is not None:
                ramp_tracker_blue.start_frame = clip_start
            print(f"[ROI] Tracker gates applied")

    # Auto-load ground-truth labels for this clip if they exist. The
    # /api/benchmark endpoint then reports live accuracy of auto vs GT.
    if args.replay:
        gt_basename = os.path.splitext(os.path.basename(args.replay))[0]
        gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "labels", gt_basename + ".json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path) as f:
                    _ground_truth = json.load(f)
                gt_red = _ground_truth["counts"]["red"]
                gt_blue = _ground_truth["counts"]["blue"]
                print(f"[GT] Loaded ground truth: labels/{gt_basename}.json")
                print(f"     R cls={gt_red['classified']} ovr={gt_red['overflow']} | "
                      f"B cls={gt_blue['classified']} ovr={gt_blue['overflow']} | "
                      f"events={len(_ground_truth.get('events', []))}")
            except (json.JSONDecodeError, KeyError, OSError) as e:
                print(f"[GT] Could not load {gt_path}: {e}")
        else:
            print(f"[GT] No labels file at {gt_path} — benchmark disabled.")

    # Determine camera mode
    if args.replay:
        cam_mode = f"Video replay: {args.replay} @ {args.replay_fps}fps (ESP32 quality)"
    elif use_usb:
        cam_mode = f"USB webcam (index {args.usb})"
    elif args.capture:
        cam_mode = f"ESP32 capture polling @ {config.ESP32_CAPTURE_URL}"
    else:
        cam_mode = f"ESP32 MJPEG stream @ {config.ESP32_STREAM_URL}"

    print("=" * 60)
    print("  FTC DECODE Scoring Vision System")
    print("=" * 60)
    print(f"  Web UI:    http://localhost:{args.port}")
    print(f"  Detector:  {det_mode}")
    print(f"  Tracker:   {'RampTracker (dual alliance)' if ramp_tracker_red else 'Direct (no tracker)'}")
    print(f"  Camera:    {cam_mode}")
    print("=" * 60)

    # Start grab thread
    if args.replay:
        grab_thread = threading.Thread(
            target=grab_loop_replay,
            args=(args.replay, args.replay_fps, not args.no_loop),
            daemon=True,
        )
    elif use_usb:
        cap = open_usb_camera(args.usb)
        if cap is None:
            import sys
            sys.exit(1)
        grab_thread = threading.Thread(target=grab_loop_usb, args=(cap,), daemon=True)
    elif args.capture:
        grab_thread = threading.Thread(target=grab_loop_capture, daemon=True)
    else:
        grab_thread = threading.Thread(target=grab_loop, daemon=True)
    grab_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(target=process_loop, daemon=True)
    proc_thread.start()

    # Match phase is manually advanced via /api/match/advance (no timer thread)

    print(f"\n[*] Starting web server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
