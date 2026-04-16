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
_roi_cache = None  # In-memory cache — avoids disk reads on every frame


def _default_roi_config():
    """Return a blank dual-alliance ROI config."""
    return {
        "red": {"roi": None, "gate": None, "exit": None, "divider": None},
        "blue": {"roi": None, "gate": None, "exit": None, "divider": None},
    }


def load_roi_config():
    """Load ROI config from memory cache, falling back to disk on first call.

    Handles migration from the old single-ROI format to dual-alliance format.
    """
    global _roi_cache
    if _roi_cache is not None:
        return _roi_cache
    if os.path.exists(ROI_CONFIG_PATH):
        try:
            with open(ROI_CONFIG_PATH, "r") as f:
                data = json.load(f)
                # Migrate old single-ROI format -> dual-alliance
                if "red" not in data and "blue" not in data:
                    migrated = _default_roi_config()
                    if data.get("roi") or data.get("gate"):
                        migrated["red"]["roi"] = data.get("roi")
                        migrated["red"]["gate"] = data.get("gate")
                    _roi_cache = migrated
                    return _roi_cache
                _roi_cache = data
                return _roi_cache
        except (json.JSONDecodeError, IOError):
            pass
    _roi_cache = _default_roi_config()
    return _roi_cache


def save_roi_config(data):
    """Save ROI and gate zone to disk and update cache."""
    global _roi_cache
    _roi_cache = data
    with open(ROI_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


STREAM_TIMEOUT = (5, 30)
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

    try:
        url = f"{config.ESP32_CONTROL_URL}?var=framesize&val={config.ESP32_DEFAULT_FRAMESIZE}"
        resp = http_requests.get(url, timeout=3)
        if resp.status_code == 200:
            print(f"    Set framesize={config.ESP32_DEFAULT_FRAMESIZE} (SVGA 800x600)")
            print("    Waiting for camera to stabilize...")
            time.sleep(2)
        else:
            print(f"    [!] framesize returned {resp.status_code}")
    except Exception as e:
        print(f"    [!] Could not set framesize: {e}")

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

        # Always push our framesize/quality settings after every (re)connect.
        # If the ESP32 rebooted — e.g. after a fall or brief power loss — it
        # comes back up in the firmware default (often lower-res/lower-quality)
        # and we need to re-assert SVGA + our quality setting. Previously
        # this ran once per process, so a post-crash reconnect silently
        # left the camera in whatever degraded state it rebooted into.
        global esp32_configured
        esp32_configured = False
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
_replay_target_fps = 15
_replay_paused = True          # Start paused so user can set up zones first
_replay_seek_to = None         # Set by API to trigger a frame seek
_replay_total_frames = 0       # Total frames in current video
_replay_current_frame = 0      # Current frame position
_replay_src_fps = 30           # Source video FPS for time calculations


def grab_loop_replay(video_path, target_fps=15, loop=True):
    """Grab loop that replays a video file at native resolution.

    Uses the video's native resolution (no downscaling) with moderate JPEG
    compression. For ESP32 simulation use --usb or the actual camera.
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
        print(f"[+] Replay: {os.path.basename(video_path)}")
        print(f"    Source: {src_w}x{src_h} @ {src_fps:.1f}fps, {total} frames")
        print(f"    Output: native res @ {_replay_target_fps}fps")

        last_time = time.time()
        grab_count = 0

        # Show first frame immediately (even while paused)
        ret, first_frame = cap.read()
        if ret:
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
            _replay_current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Encode as JPEG for the raw feed (no downscaling)
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


def _process_alliance_roi_crop(frame, roi_poly, tracker, scorer_obj, detector_ref, stream_id="default",
                                alliance_roi_data=None):
    """Crop bounding box of an alliance ROI polygon, upscale, detect, map back.

    roi_poly: [[x,y], ...] normalized polygon (3+ points).
    Returns: (mapped_balls, stable_pattern, raw_pattern, annotated_crop)
    annotated_crop is the 640x640 upscaled crop with detection boxes drawn, or None.
    """
    h, w = frame.shape[:2]
    poly_px = _poly_to_pixels(roi_poly, w, h)

    # Get bounding box of polygon
    bx, by, bw, bh = cv2.boundingRect(poly_px)
    bx2, by2 = bx + bw, by + bh
    bx, by = max(0, bx), max(0, by)
    bx2, by2 = min(w, bx2), min(h, by2)
    crop_w, crop_h = max(bx2 - bx, 1), max(by2 - by, 1)

    crop = frame[by:by2, bx:bx2]
    if crop.size == 0:
        return [], "", "", None

    # Enhancement pipeline (all optional):
    #   1. FSRCNN 2× super-resolution on the raw crop (if small enough)
    #   2. Cubic interpolation to 640x640 (vs linear by default)
    #   3. Unsharp mask for edge contrast
    #   4. CLAHE on luminance for local contrast
    # Skipped entirely when `_enhance_roi_enabled` is False.
    if _enhance_roi_enabled:
        crop_for_resize = _maybe_sr_upscale(crop)
        upscaled = cv2.resize(crop_for_resize, (640, 640),
                              interpolation=cv2.INTER_CUBIC)
        upscaled = _enhance_image(upscaled)
    else:
        upscaled = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
    annotated = upscaled.copy()
    # Stash the crop→640 mapping so we can draw alliance zones on the
    # enhanced ROI feed after detection runs.
    _crop_ann_meta = {
        "bx": bx, "by": by, "crop_w": crop_w, "crop_h": crop_h,
    }

    try:
        balls, stable_pattern, raw_pattern, masks = detector_ref.detect(upscaled, stream_id=stream_id)
    except TypeError:
        balls, stable_pattern, raw_pattern, masks = detector_ref.detect(upscaled)

    # Draw detection boxes on the annotated crop (in crop-space, before remapping)
    for b in balls:
        cx = int(b.get("x", 0))
        cy = int(b.get("y", 0))
        cwb = int(b.get("w", 0))
        chb = int(b.get("h", 0))
        color = (0, 255, 0) if b.get("color") == "G" else (200, 0, 200)
        cv2.rectangle(annotated, (cx, cy), (cx + cwb, cy + chb), color, 2)
        conf = b.get("confidence", 0)
        cv2.putText(annotated, f"{b.get('color','?')} {conf:.2f}",
                    (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Map detection coordinates back to full-frame space
    scale_x = crop_w / 640.0
    scale_y = crop_h / 640.0
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

    # Feed tracker for total count / classified-vs-overflow state
    if tracker is not None:
        tracker.update(mapped, (h, w))

    # Build the colors list for MOTIF scoring by snapshotting the CURRENT
    # balls on the RAMP, sorted along the ramp direction (from gate outward).
    # This means each position on the ramp is compared to motif[i] regardless
    # of order-of-entry. If a ball is undetected in a given frame the slot
    # will briefly be absent — but next frame it'll reappear.
    ramp_colors = _sort_balls_along_ramp(mapped, roi_poly, tracker, w, h)
    if tracker is not None:
        totals = tracker.get_totals()
        scorer_obj.update(
            ramp_colors,
            classified_total=totals["classified"],
            overflow_total=totals["overflow"],
        )
    else:
        scorer_obj.update(ramp_colors)

    # Overlay alliance zones (ROI/gate/exit/divider) on the enhanced ROI
    # feed, plus a tiny color-key bar at the bottom.
    if alliance_roi_data:
        meta = {
            "bx": _crop_ann_meta["bx"],
            "by": _crop_ann_meta["by"],
            "crop_w": _crop_ann_meta["crop_w"],
            "crop_h": _crop_ann_meta["crop_h"],
            "full_w": w,
            "full_h": h,
        }
        _draw_roi_feed_zones(annotated, alliance_roi_data, meta)

    return mapped, stable_pattern, raw_pattern, annotated


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


def process_loop():
    """Processing thread: picks up the latest grabbed frame, runs detection for both alliances."""
    global latest_processed_jpg, latest_red_roi_jpg, latest_blue_roi_jpg
    global latest_balls, latest_stable_pattern, latest_raw_pattern

    last_seq = -1
    last_time = time.time()
    proc_count = 0

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

            for alliance, tracker, scorer_obj in [
                ("red", ramp_tracker_red, scorer_red),
                ("blue", ramp_tracker_blue, scorer_blue),
            ]:
                a_roi = (roi_data.get(alliance) or {}).get("roi")
                if a_roi is None:
                    scorer_obj.update([])
                    continue

                mapped_balls, s_pat, r_pat, annot_crop = _process_alliance_roi_crop(
                    frame, a_roi, tracker, scorer_obj, detector,
                    stream_id=alliance,
                    alliance_roi_data=roi_data.get(alliance) or {},
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
                    tracker.update(alliance_balls, (h, w))
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
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
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
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
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
        red_snap = scorer_red.get_scores()
        blue_snap = scorer_blue.get_scores()
        # Freeze balls currently on the ramp so they carry over into TELEOP
        # without being recounted (handles track_id reassignment during the
        # AUTO-TELEOP pause).
        if ramp_tracker_red is not None:
            ramp_tracker_red.handoff_phase()
        if ramp_tracker_blue is not None:
            ramp_tracker_blue.handoff_phase()
        with match_state_lock:
            match_state["auto_snapshot"]["red"] = red_snap
            match_state["auto_snapshot"]["blue"] = blue_snap
            match_state["phase"] = "TELEOP"
            match_state["started_at"] = time.time()
        print("[MATCH] AUTO ended (manual), TELEOP started")
    elif current == "TELEOP":
        red_snap = scorer_red.get_scores()
        blue_snap = scorer_blue.get_scores()
        with match_state_lock:
            match_state["final_snapshot"]["red"] = red_snap
            match_state["final_snapshot"]["blue"] = blue_snap
            match_state["phase"] = "ENDED"
            match_state["started_at"] = None
        _replay_paused = True
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


@app.route("/api/yolo_config", methods=["GET"])
def api_yolo_config():
    """Return current YOLO config for UI display."""
    return jsonify({
        "confidence": config.YOLO_CONFIDENCE,
        "iou_threshold": config.YOLO_IOU_THRESHOLD,
        "model_path": config.YOLO_MODEL_PATH,
    })


@app.route("/api/yolo_config", methods=["POST"])
def api_set_yolo_config():
    """Update YOLO config live."""
    data = request.get_json(silent=True) or {}

    if "confidence" in data:
        config.YOLO_CONFIDENCE = float(data["confidence"])
        if hasattr(detector, 'confidence'):
            detector.confidence = config.YOLO_CONFIDENCE
    if "iou_threshold" in data:
        config.YOLO_IOU_THRESHOLD = float(data["iou_threshold"])
        if hasattr(detector, 'iou_threshold'):
            detector.iou_threshold = config.YOLO_IOU_THRESHOLD

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
            save_data[alliance] = {"roi": None, "gate": None, "exit": None, "divider": None}
        if "roi" in a_data:
            save_data[alliance]["roi"] = a_data["roi"]
            if tracker is not None and a_data["roi"] is not None:
                tracker.set_roi(a_data["roi"])
        if "gate" in a_data:
            save_data[alliance]["gate"] = a_data["gate"]
            if tracker is not None and a_data["gate"] is not None:
                tracker.set_gate_zone(a_data["gate"])
        if "exit" in a_data:
            save_data[alliance]["exit"] = a_data["exit"]
            if tracker is not None:
                tracker.set_exit_zone(a_data["exit"])
        if "divider" in a_data:
            save_data[alliance]["divider"] = a_data["divider"]
            # Propagate to tracker if it supports a divider (optional method).
            if tracker is not None and hasattr(tracker, "set_divider"):
                tracker.set_divider(a_data["divider"])

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
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
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
    if hasattr(detector, 'reset_tracker'):
        detector.reset_tracker()
    scorer_red.update([])
    scorer_blue.update([])

    _replay_switch_to = path
    return jsonify({"status": "ok", "playing": name})


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
    """Get replay state including seek position."""
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
    })


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
    parser.add_argument("--replay-fps", type=int, default=15,
                        help="Replay FPS (default: 15, similar to ESP32 stream)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Don't loop the replay video")
    parser.add_argument("--live", action="store_true",
                        help="Force live ESP32-S3 camera (clears --replay/--usb). "
                             "Same as default ESP32 path, but explicit.")
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

    if args.live:
        # Make intent unambiguous: ignore replay/usb if accidentally passed alongside.
        args.replay = None
        args.usb = None
        print("[LIVE] ESP32-S3 camera mode requested — connect to the AP at "
              f"{config.ESP32_STREAM_URL.split('/stream')[0]} first.")

    use_usb = args.usb is not None

    # Override stream URL if provided
    if args.stream_url:
        config.ESP32_STREAM_URL = args.stream_url

    # Initialize detector (hybrid HSV+YOLO cascade, or pure YOLO)
    try:
        model_path = args.yolo_model or config.YOLO_MODEL_PATH
        from yolo_detector import YOLODetector
        detector = YOLODetector(model_path=model_path)
        det_mode = f"YOLO ({model_path})"
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

    # Initialize dual ramp trackers
    try:
        if args.tracker == "simple":
            from simple_count_tracker import SimpleCountTracker as _Tracker
            tracker_mode = "SIMPLE (smoothed live count)"
        else:
            from ramp_tracker import RampTracker as _Tracker
            tracker_mode = "FULL (cumulative classified/overflow)"
        ramp_tracker_red = _Tracker()
        ramp_tracker_blue = _Tracker()
        print(f"  Tracker: {tracker_mode}")
        # Load saved ROI/gate config for each alliance
        roi_data = load_roi_config()
        for alliance, tracker in [("red", ramp_tracker_red), ("blue", ramp_tracker_blue)]:
            a_data = roi_data.get(alliance, {})
            if a_data.get("roi"):
                tracker.set_roi(a_data["roi"])
                print(f"  Loaded {alliance} ROI: {a_data['roi']}")
            if a_data.get("gate"):
                tracker.set_gate_zone(a_data["gate"])
                print(f"  Loaded {alliance} Gate: {a_data['gate']}")
            if a_data.get("exit"):
                tracker.set_exit_zone(a_data["exit"])
                print(f"  Loaded {alliance} Exit: {a_data['exit']}")
            if a_data.get("divider") and hasattr(tracker, "set_divider"):
                tracker.set_divider(a_data["divider"])
                print(f"  Loaded {alliance} Divider: {a_data['divider']}")
    except ImportError:
        print("[i] ramp_tracker module not found — using direct detection mode")
        ramp_tracker_red = None
        ramp_tracker_blue = None

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
