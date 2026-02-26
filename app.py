"""
FTC DECODE Scoring System - Web Application
Flask server that captures ESP32-CAM feed, runs vision detection, and serves a dashboard.

Usage:
    python app.py                    # Connect to ESP32-CAM over WiFi
    python app.py --usb              # Use USB webcam (index 0)
    python app.py --usb 2            # Use USB webcam at index 2
    python app.py --yolo             # Use YOLOv8 detector (default model path)
    python app.py --yolo --yolo-model path/to/best.pt
    python app.py --port 9000        # Use a different port
"""

import argparse
import threading
import time

import cv2
import numpy as np
import requests as http_requests
from flask import Flask, Response, jsonify, render_template, request

import config
from detector import BallDetector
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
latest_green_mask_jpg = None  # Pre-encoded JPEG bytes of green mask
latest_purple_mask_jpg = None # Pre-encoded JPEG bytes of purple mask
latest_balls = []
latest_stable_pattern = ""
latest_raw_pattern = ""
fps_counter = {"grab": 0.0, "process": 0.0}

JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 80]

detector = None  # Initialized in __main__ after arg parsing
scorer = ScoreKeeper()

# Snapshot defaults at startup so "Reset" always has the originals
CONFIG_DEFAULTS = {
    "green_hsv_lower": config.GREEN_HSV_LOWER.tolist(),
    "green_hsv_upper": config.GREEN_HSV_UPPER.tolist(),
    "green_ycrcb_lower": config.GREEN_YCRCB_LOWER.tolist(),
    "green_ycrcb_upper": config.GREEN_YCRCB_UPPER.tolist(),
    "purple_hsv_lower": config.PURPLE_HSV_LOWER.tolist(),
    "purple_hsv_upper": config.PURPLE_HSV_UPPER.tolist(),
    "purple_ycrcb_lower": config.PURPLE_YCRCB_LOWER.tolist(),
    "purple_ycrcb_upper": config.PURPLE_YCRCB_UPPER.tolist(),
    "min_area": config.MIN_CONTOUR_AREA,
    "max_area": config.MAX_CONTOUR_AREA,
    "min_circularity": config.MIN_CIRCULARITY,
    "min_solidity": config.MIN_SOLIDITY,
}


# ============= Auto-Tune State =============
autotune_state = {"running": False, "progress": 0, "message": "", "result": None}


def run_autotune(roi_norm, target_color):
    """Auto-tune color thresholds using full detection pipeline scoring.

    Instead of optimizing pixel-level thresholds, this runs the actual
    detection pipeline (threshold → morphology → contours → filtering)
    and optimizes for: "detect ball-shaped contours inside ROI consistently,
    detect nothing outside ROI."
    """
    global autotune_state
    autotune_state = {"running": True, "progress": 0,
                      "message": "Collecting frames...", "result": None}

    try:
        _autotune_inner(roi_norm, target_color)
    except Exception as e:
        autotune_state = {"running": False, "progress": 0,
                          "message": f"Error: {e}", "result": None}


def _detect_balls_single_color(hsv_img, ycrcb_img, hsv_lo, hsv_hi,
                                ycrcb_lo, ycrcb_hi, kernel, kernel_large):
    """Run full detection pipeline for one color. Returns list of (cx, cy, area)."""
    mask_hsv = cv2.inRange(hsv_img, hsv_lo, hsv_hi)
    mask_ycrcb = cv2.inRange(ycrcb_img, ycrcb_lo, ycrcb_hi)
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                            iterations=config.MORPH_OPEN_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large,
                            iterations=config.MORPH_CLOSE_ITERATIONS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter * perimeter)
        if circ < config.MIN_CIRCULARITY:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            continue
        if area / hull_area < config.MIN_SOLIDITY:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((x + w // 2, y + h // 2, area))

    return detections


def _autotune_inner(roi_norm, target_color):
    global autotune_state

    # Phase 1: Collect frames
    frames = []
    last_seq = -1
    for _ in range(300):
        with frame_lock:
            if frame_seq != last_seq and latest_frame is not None:
                frames.append(latest_frame.copy())
                last_seq = frame_seq
        if len(frames) >= 15:
            break
        time.sleep(0.1)

    if len(frames) < 5:
        autotune_state = {"running": False, "progress": 0,
                          "message": "Not enough frames captured", "result": None}
        return

    autotune_state["progress"] = 5
    autotune_state["message"] = f"Captured {len(frames)} frames, pre-processing..."

    # Convert ROI to pixel coordinates
    fh, fw = frames[0].shape[:2]
    rx1 = max(0, int(roi_norm[0] * fw))
    ry1 = max(0, int(roi_norm[1] * fh))
    rx2 = min(fw, int(roi_norm[2] * fw))
    ry2 = min(fh, int(roi_norm[3] * fh))

    # Pre-compute color spaces (done once, reused for every evaluation)
    all_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
    all_ycrcb = [cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) for f in frames]

    # Pre-allocate morphology kernels
    kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
    kernel_large = np.ones((config.MORPH_KERNEL_LARGE, config.MORPH_KERNEL_LARGE), np.uint8)

    autotune_state["progress"] = 15

    # Phase 2: Start from CURRENT config values (which already partially work)
    # The user has already done manual tuning — we refine from there,
    # not from scratch.
    if target_color == "green":
        hsv_lo = config.GREEN_HSV_LOWER.astype(int).copy()
        hsv_hi = config.GREEN_HSV_UPPER.astype(int).copy()
        ycrcb_lo = config.GREEN_YCRCB_LOWER.astype(int).copy()
        ycrcb_hi = config.GREEN_YCRCB_UPPER.astype(int).copy()
    else:
        hsv_lo = config.PURPLE_HSV_LOWER.astype(int).copy()
        hsv_hi = config.PURPLE_HSV_UPPER.astype(int).copy()
        ycrcb_lo = config.PURPLE_YCRCB_LOWER.astype(int).copy()
        ycrcb_hi = config.PURPLE_YCRCB_UPPER.astype(int).copy()

    autotune_state["progress"] = 20
    n_frames = len(frames)
    roi_pixel_count = max((rx2 - rx1) * (ry2 - ry1), 1)
    outside_pixel_count = max(fh * fw - roi_pixel_count, 1)

    # Phase 3: Hybrid scoring — mask pixel coverage (continuous gradient)
    # plus contour detection bonus (the actual goal).
    # Mask pixel count gives the optimizer a direction to follow even
    # when no contours are detected yet.
    eval_count = [0]

    def score(h_lo, h_hi, y_lo, y_hi):
        eval_count[0] += 1
        h_lo_arr = np.array(h_lo, dtype=np.uint8)
        h_hi_arr = np.array(h_hi, dtype=np.uint8)
        y_lo_arr = np.array(y_lo, dtype=np.uint8)
        y_hi_arr = np.array(y_hi, dtype=np.uint8)

        total_mask_inside = 0
        total_mask_outside = 0
        frames_with_contour = 0

        for hsv, ycrcb in zip(all_hsv, all_ycrcb):
            # Raw mask (before morphology) — continuous signal
            mask_hsv = cv2.inRange(hsv, h_lo_arr, h_hi_arr)
            mask_ycrcb = cv2.inRange(ycrcb, y_lo_arr, y_hi_arr)
            raw_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

            roi_mask = raw_mask[ry1:ry2, rx1:rx2]
            total_mask_inside += cv2.countNonZero(roi_mask)

            outside = raw_mask.copy()
            outside[ry1:ry2, rx1:rx2] = 0
            total_mask_outside += cv2.countNonZero(outside)

            # Full pipeline — contour detection bonus
            dets = _detect_balls_single_color(
                hsv, ycrcb, h_lo_arr, h_hi_arr, y_lo_arr, y_hi_arr,
                kernel, kernel_large)

            for cx, cy, area in dets:
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    frames_with_contour += 1
                    break  # one hit per frame is enough

        # Mask pixel coverage (0-1 range, provides gradient)
        inside_coverage = total_mask_inside / (roi_pixel_count * n_frames)
        outside_coverage = total_mask_outside / (outside_pixel_count * n_frames)

        # Contour detection rate (0-1 range, the real goal)
        contour_rate = frames_with_contour / n_frames

        # Hybrid: follow the pixel gradient, reward contour detections,
        # heavily penalize outside leakage
        return (0.3 * inside_coverage
                + 0.7 * contour_rate
                - 3.0 * outside_coverage)

    best_h_lo = hsv_lo.copy()
    best_h_hi = hsv_hi.copy()
    best_y_lo = ycrcb_lo.copy()
    best_y_hi = ycrcb_hi.copy()
    best_score = score(best_h_lo, best_h_hi, best_y_lo, best_y_hi)

    autotune_state["progress"] = 25
    autotune_state["message"] = f"Initial score: {best_score:.3f}, optimizing..."

    # Phase 4: Coordinate descent with full pipeline scoring
    steps = [12, 6, 3, 1]
    for si, step in enumerate(steps):
        autotune_state["progress"] = 25 + int(65 * (si / len(steps)))
        autotune_state["message"] = (
            f"Step {step} — score: {best_score:.3f} "
            f"({eval_count[0]} evals)")

        improved = True
        iters = 0
        while improved and iters < 40:
            improved = False
            iters += 1

            for arr, idx, max_val in [
                (best_h_lo, 0, 179), (best_h_lo, 1, 255), (best_h_lo, 2, 255),
                (best_h_hi, 0, 179), (best_h_hi, 1, 255), (best_h_hi, 2, 255),
                (best_y_lo, 0, 255), (best_y_lo, 1, 255), (best_y_lo, 2, 255),
                (best_y_hi, 0, 255), (best_y_hi, 1, 255), (best_y_hi, 2, 255),
            ]:
                original = arr[idx]
                best_delta = 0

                for delta in [step, -step]:
                    new_val = max(0, min(int(original) + delta, max_val))
                    if new_val == original:
                        continue
                    arr[idx] = new_val
                    s = score(best_h_lo, best_h_hi, best_y_lo, best_y_hi)
                    if s > best_score + 0.001:
                        best_score = s
                        best_delta = delta
                        improved = True

                if best_delta != 0:
                    arr[idx] = max(0, min(int(original) + best_delta, max_val))
                else:
                    arr[idx] = original

    # Apply optimized thresholds
    final_hsv_lo = np.array(best_h_lo, dtype=np.uint8)
    final_hsv_hi = np.array(best_h_hi, dtype=np.uint8)
    final_ycrcb_lo = np.array(best_y_lo, dtype=np.uint8)
    final_ycrcb_hi = np.array(best_y_hi, dtype=np.uint8)

    if target_color == "green":
        config.GREEN_HSV_LOWER = final_hsv_lo
        config.GREEN_HSV_UPPER = final_hsv_hi
        config.GREEN_YCRCB_LOWER = final_ycrcb_lo
        config.GREEN_YCRCB_UPPER = final_ycrcb_hi
    else:
        config.PURPLE_HSV_LOWER = final_hsv_lo
        config.PURPLE_HSV_UPPER = final_hsv_hi
        config.PURPLE_YCRCB_LOWER = final_ycrcb_lo
        config.PURPLE_YCRCB_UPPER = final_ycrcb_hi

    # Get final detection stats
    final_s = score(best_h_lo, best_h_hi, best_y_lo, best_y_hi)
    h_lo_arr = np.array(best_h_lo, dtype=np.uint8)
    h_hi_arr = np.array(best_h_hi, dtype=np.uint8)
    y_lo_arr = np.array(best_y_lo, dtype=np.uint8)
    y_hi_arr = np.array(best_y_hi, dtype=np.uint8)

    final_inside = 0
    final_outside = 0
    final_frames_hit = 0
    for hsv, ycrcb in zip(all_hsv, all_ycrcb):
        dets = _detect_balls_single_color(
            hsv, ycrcb, h_lo_arr, h_hi_arr, y_lo_arr, y_hi_arr,
            kernel, kernel_large)
        fi = sum(1 for cx, cy, _ in dets if rx1 <= cx <= rx2 and ry1 <= cy <= ry2)
        fo = len(dets) - fi
        final_inside += fi
        final_outside += fo
        if fi > 0:
            final_frames_hit += 1

    autotune_state = {
        "running": False, "progress": 100,
        "message": (f"Done! Detected in {final_frames_hit}/{n_frames} frames, "
                    f"{final_inside} inside / {final_outside} outside ROI"),
        "result": {
            "color": target_color,
            "hsv_lower": final_hsv_lo.tolist(),
            "hsv_upper": final_hsv_hi.tolist(),
            "ycrcb_lower": final_ycrcb_lo.tolist(),
            "ycrcb_upper": final_ycrcb_hi.tolist(),
            "score": round(best_score, 4),
            "frames_detected": final_frames_hit,
            "total_frames": n_frames,
            "inside_detections": final_inside,
            "outside_detections": final_outside,
            "evaluations": eval_count[0],
        },
    }


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


def configure_esp32_camera():
    """Set resolution and quality on the ESP32 camera. Only runs once.

    Sends /control?var=framesize&val=N (SVGA=11 on Freenove S3).
    If /control fails, tries /resolution endpoint as fallback.
    Waits 2s after changing resolution for the camera to stabilize.
    """
    global esp32_configured
    if esp32_configured:
        return

    # Try /control endpoint (works on both old ESP32-CAM and Freenove S3)
    try:
        # Set quality first (less disruptive)
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

    # Set framesize (this restarts the stream briefly)
    try:
        url = f"{config.ESP32_CONTROL_URL}?var=framesize&val={config.ESP32_DEFAULT_FRAMESIZE}"
        resp = http_requests.get(url, timeout=3)
        if resp.status_code == 200:
            print(f"    Set framesize={config.ESP32_DEFAULT_FRAMESIZE} (SVGA 800x600)")
            print("    Waiting for camera to stabilize...")
            time.sleep(2)  # Give camera time to restart stream at new resolution
        else:
            print(f"    [!] framesize returned {resp.status_code}")
    except Exception as e:
        print(f"    [!] Could not set framesize: {e}")

    esp32_configured = True


def grab_loop():
    """Fast thread: reads raw MJPEG stream, parses JPEG frames directly.

    After first successful connection, configures camera resolution.
    Tracks all disconnects and FPS for the connection monitor.
    """
    global latest_frame, latest_raw_jpg, frame_seq

    first_connect = True

    while True:
        # Connect
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

        # Configure camera AFTER first successful stream connection
        if first_connect:
            first_connect = False
            # Close the test stream — configure will briefly disrupt it
            try:
                stream.close()
            except Exception:
                pass
            configure_esp32_camera()
            # Reconnect after configuration
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
                    if grab_count > 0:
                        conn_monitor.record_fps(current_fps)
                    grab_count = 0
                    last_time = time.time()
                    conn_monitor.maybe_print_report()

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
    """Grab loop for USB webcam. Takes an already-opened VideoCapture.
    Same contract as grab_loop — writes to latest_frame, latest_raw_jpg,
    frame_seq under frame_lock.
    """
    global latest_frame, latest_raw_jpg, frame_seq

    last_time = time.time()
    grab_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] USB webcam read failed, retrying...")
            time.sleep(0.1)
            continue

        # Encode to JPEG for the raw feed
        _, jpg_buf = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
        jpg_bytes = jpg_buf.tobytes()

        with frame_lock:
            latest_frame = frame
            latest_raw_jpg = jpg_bytes
            frame_seq += 1

        grab_count += 1

        # FPS tracking
        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_counter["grab"] = grab_count / elapsed
            grab_count = 0
            last_time = time.time()


def grab_loop_capture():
    """Grab loop using /capture endpoint (single JPEG polling).

    More compatible than MJPEG streaming — works with Freenove ESP32-S3
    and doesn't require port 81. Slightly lower FPS but very reliable.
    """
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

            with frame_lock:
                latest_frame = frame
                latest_raw_jpg = jpg_bytes
                frame_seq += 1

            grab_count += 1

        except Exception as e:
            print(f"[!] Capture error: {e}")
            time.sleep(1)
            continue

        # FPS tracking
        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_counter["grab"] = grab_count / elapsed
            grab_count = 0
            last_time = time.time()


def process_loop():
    """Processing thread: picks up the latest grabbed frame, runs detection.

    Pre-encodes output JPEGs so Flask generators do zero OpenCV work.
    """
    global latest_processed_jpg, latest_green_mask_jpg, latest_purple_mask_jpg
    global latest_balls, latest_stable_pattern, latest_raw_pattern

    last_seq = -1
    last_time = time.time()
    proc_count = 0

    while True:
        # Grab the latest frame
        with frame_lock:
            frame = latest_frame
            seq = frame_seq

        # No new frame yet — wait briefly
        if frame is None or seq == last_seq:
            time.sleep(0.005)
            continue

        last_seq = seq

        # Run detection
        balls, stable_pattern, raw_pattern, masks = detector.detect(frame)
        processed = detector.draw_detections(frame, balls, stable_pattern, raw_pattern)

        # Pre-encode all output JPEGs here (not in generator threads)
        _, proc_buf = cv2.imencode(".jpg", processed, JPEG_ENCODE_PARAMS)
        proc_jpg = proc_buf.tobytes()

        green_jpg = None
        purple_jpg = None
        green_mask = masks.get("green")
        purple_mask = masks.get("purple")
        if green_mask is not None:
            _, g_buf = cv2.imencode(".jpg", cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR), JPEG_ENCODE_PARAMS)
            green_jpg = g_buf.tobytes()
        if purple_mask is not None:
            _, p_buf = cv2.imencode(".jpg", cv2.cvtColor(purple_mask, cv2.COLOR_GRAY2BGR), JPEG_ENCODE_PARAMS)
            purple_jpg = p_buf.tobytes()

        # Update scorer
        detected_colors = [b["color"] for b in balls]
        scorer.update(detected_colors)

        # Publish pre-encoded results (lock held very briefly — just pointer swaps)
        with output_lock:
            latest_processed_jpg = proc_jpg
            latest_green_mask_jpg = green_jpg
            latest_purple_mask_jpg = purple_jpg
            latest_balls = balls
            latest_stable_pattern = stable_pattern
            latest_raw_pattern = raw_pattern

        # FPS tracking
        proc_count += 1
        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_counter["process"] = proc_count / elapsed
            proc_count = 0
            last_time = time.time()


_blank_jpg = None  # Cached "waiting" frame


def _get_blank_jpg():
    global _blank_jpg
    if _blank_jpg is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        _, buf = cv2.imencode(".jpg", blank, JPEG_ENCODE_PARAMS)
        _blank_jpg = buf.tobytes()
    return _blank_jpg


def generate_mjpeg(feed_type="processed"):
    """Generator that yields pre-encoded JPEG frames. Zero OpenCV work here."""
    while True:
        jpg_bytes = None

        if feed_type == "raw":
            with frame_lock:
                jpg_bytes = latest_raw_jpg
        else:
            with output_lock:
                if feed_type == "green_mask":
                    jpg_bytes = latest_green_mask_jpg
                elif feed_type == "purple_mask":
                    jpg_bytes = latest_purple_mask_jpg
                else:
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


@app.route("/video_feed/green_mask")
def video_feed_green_mask():
    return Response(generate_mjpeg("green_mask"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/purple_mask")
def video_feed_purple_mask():
    return Response(generate_mjpeg("purple_mask"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/scores")
def api_scores():
    """Return current scoring data as JSON."""
    scores = scorer.get_scores()
    with output_lock:
        scores["raw_pattern"] = latest_raw_pattern
        scores["stable_pattern"] = latest_stable_pattern
        scores["ball_count"] = len(latest_balls)
    scores["fps"] = round(fps_counter.get("process", 0), 1)
    scores["grab_fps"] = round(fps_counter.get("grab", 0), 1)
    return jsonify(scores)


@app.route("/api/motif", methods=["POST"])
def api_set_motif():
    """Set the active MOTIF."""
    data = request.get_json(silent=True) or {}
    motif = data.get("motif", "").upper()
    if motif in config.MOTIFS:
        scorer.set_motif(motif)
        return jsonify({"status": "ok", "motif": motif})
    return jsonify({"status": "error", "message": f"Invalid motif: {motif}"}), 400


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


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return current detection config for UI display."""
    return jsonify({
        "green_hsv": {
            "lower": config.GREEN_HSV_LOWER.tolist(),
            "upper": config.GREEN_HSV_UPPER.tolist(),
        },
        "green_ycrcb": {
            "lower": config.GREEN_YCRCB_LOWER.tolist(),
            "upper": config.GREEN_YCRCB_UPPER.tolist(),
        },
        "purple_hsv": {
            "lower": config.PURPLE_HSV_LOWER.tolist(),
            "upper": config.PURPLE_HSV_UPPER.tolist(),
        },
        "purple_ycrcb": {
            "lower": config.PURPLE_YCRCB_LOWER.tolist(),
            "upper": config.PURPLE_YCRCB_UPPER.tolist(),
        },
        "min_area": config.MIN_CONTOUR_AREA,
        "max_area": config.MAX_CONTOUR_AREA,
        "min_circularity": config.MIN_CIRCULARITY,
        "min_solidity": config.MIN_SOLIDITY,
    })


@app.route("/api/config", methods=["POST"])
def api_set_config():
    """Update detection config live. Accepts partial updates."""
    data = request.get_json(silent=True) or {}

    if "green_hsv" in data:
        g = data["green_hsv"]
        if "lower" in g:
            config.GREEN_HSV_LOWER = np.array(g["lower"], dtype=np.uint8)
        if "upper" in g:
            config.GREEN_HSV_UPPER = np.array(g["upper"], dtype=np.uint8)

    if "green_ycrcb" in data:
        g = data["green_ycrcb"]
        if "lower" in g:
            config.GREEN_YCRCB_LOWER = np.array(g["lower"], dtype=np.uint8)
        if "upper" in g:
            config.GREEN_YCRCB_UPPER = np.array(g["upper"], dtype=np.uint8)

    if "purple_hsv" in data:
        p = data["purple_hsv"]
        if "lower" in p:
            config.PURPLE_HSV_LOWER = np.array(p["lower"], dtype=np.uint8)
        if "upper" in p:
            config.PURPLE_HSV_UPPER = np.array(p["upper"], dtype=np.uint8)

    if "purple_ycrcb" in data:
        p = data["purple_ycrcb"]
        if "lower" in p:
            config.PURPLE_YCRCB_LOWER = np.array(p["lower"], dtype=np.uint8)
        if "upper" in p:
            config.PURPLE_YCRCB_UPPER = np.array(p["upper"], dtype=np.uint8)

    if "min_area" in data:
        config.MIN_CONTOUR_AREA = int(data["min_area"])
    if "max_area" in data:
        config.MAX_CONTOUR_AREA = int(data["max_area"])
    if "min_circularity" in data:
        config.MIN_CIRCULARITY = float(data["min_circularity"])
    if "min_solidity" in data:
        config.MIN_SOLIDITY = float(data["min_solidity"])

    return jsonify({"status": "ok"})


@app.route("/api/pixel_info", methods=["POST"])
def api_pixel_info():
    """Return color values at a specific pixel (x%, y%) of the current frame.

    Accepts normalized coordinates (0.0-1.0) so it works regardless of display size.
    Returns BGR, HSV, and YCrCb values at that pixel.
    """
    data = request.get_json(silent=True) or {}
    nx = float(data.get("x", 0.5))
    ny = float(data.get("y", 0.5))

    with frame_lock:
        frame = latest_frame

    if frame is None:
        return jsonify({"status": "error", "message": "No frame available"}), 503

    h, w = frame.shape[:2]
    px = min(max(int(nx * w), 0), w - 1)
    py = min(max(int(ny * h), 0), h - 1)

    bgr = frame[py, px].tolist()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px].tolist()
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[py, px].tolist()

    return jsonify({
        "px": px, "py": py,
        "bgr": bgr,
        "rgb": [bgr[2], bgr[1], bgr[0]],
        "hsv": hsv,
        "ycrcb": ycrcb,
    })


@app.route("/api/save_config", methods=["POST"])
def api_save_config():
    """Write current detection thresholds back to config.py for persistence."""
    import os

    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    try:
        with open(config_path, "r") as f:
            lines = f.read()

        # Replace threshold values in the config file
        replacements = {
            "GREEN_HSV_LOWER": config.GREEN_HSV_LOWER.tolist(),
            "GREEN_HSV_UPPER": config.GREEN_HSV_UPPER.tolist(),
            "GREEN_YCRCB_LOWER": config.GREEN_YCRCB_LOWER.tolist(),
            "GREEN_YCRCB_UPPER": config.GREEN_YCRCB_UPPER.tolist(),
            "PURPLE_HSV_LOWER": config.PURPLE_HSV_LOWER.tolist(),
            "PURPLE_HSV_UPPER": config.PURPLE_HSV_UPPER.tolist(),
            "PURPLE_YCRCB_LOWER": config.PURPLE_YCRCB_LOWER.tolist(),
            "PURPLE_YCRCB_UPPER": config.PURPLE_YCRCB_UPPER.tolist(),
        }

        import re
        for var_name, values in replacements.items():
            pattern = rf"{var_name}\s*=\s*np\.array\(\[.*?\]\)"
            replacement = f"{var_name} = np.array({values})"
            lines = re.sub(pattern, replacement, lines)

        # Replace scalar params
        scalar_replacements = {
            "MIN_CONTOUR_AREA": config.MIN_CONTOUR_AREA,
            "MAX_CONTOUR_AREA": config.MAX_CONTOUR_AREA,
            "MIN_CIRCULARITY": config.MIN_CIRCULARITY,
            "MIN_SOLIDITY": config.MIN_SOLIDITY,
        }

        for var_name, value in scalar_replacements.items():
            pattern = rf"{var_name}\s*=\s*[\d.]+"
            replacement = f"{var_name} = {value}"
            lines = re.sub(pattern, replacement, lines)

        with open(config_path, "w") as f:
            f.write(lines)

        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/reset_config", methods=["POST"])
def api_reset_config():
    """Reset all detection config to the defaults captured at startup."""
    d = CONFIG_DEFAULTS
    config.GREEN_HSV_LOWER = np.array(d["green_hsv_lower"], dtype=np.uint8)
    config.GREEN_HSV_UPPER = np.array(d["green_hsv_upper"], dtype=np.uint8)
    config.GREEN_YCRCB_LOWER = np.array(d["green_ycrcb_lower"], dtype=np.uint8)
    config.GREEN_YCRCB_UPPER = np.array(d["green_ycrcb_upper"], dtype=np.uint8)
    config.PURPLE_HSV_LOWER = np.array(d["purple_hsv_lower"], dtype=np.uint8)
    config.PURPLE_HSV_UPPER = np.array(d["purple_hsv_upper"], dtype=np.uint8)
    config.PURPLE_YCRCB_LOWER = np.array(d["purple_ycrcb_lower"], dtype=np.uint8)
    config.PURPLE_YCRCB_UPPER = np.array(d["purple_ycrcb_upper"], dtype=np.uint8)
    config.MIN_CONTOUR_AREA = d["min_area"]
    config.MAX_CONTOUR_AREA = d["max_area"]
    config.MIN_CIRCULARITY = d["min_circularity"]
    config.MIN_SOLIDITY = d["min_solidity"]
    return jsonify({"status": "ok"})


@app.route("/api/autotune", methods=["POST"])
def api_autotune():
    """Start auto-tuning thresholds for a color within a selected ROI."""
    if autotune_state.get("running"):
        return jsonify({"status": "error", "message": "Already running"}), 409

    data = request.get_json(silent=True) or {}
    roi = data.get("roi")  # [x1, y1, x2, y2] normalized 0-1
    color = data.get("color", "").lower()

    if not roi or len(roi) != 4:
        return jsonify({"status": "error", "message": "Missing ROI"}), 400
    if color not in ("green", "purple"):
        return jsonify({"status": "error", "message": "Color must be green or purple"}), 400

    t = threading.Thread(target=run_autotune, args=(roi, color), daemon=True)
    t.start()
    return jsonify({"status": "ok", "message": "Auto-tune started"})


@app.route("/api/autotune/status")
def api_autotune_status():
    """Poll auto-tune progress."""
    return jsonify(autotune_state)


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
    parser.add_argument("--yolo", action="store_true",
                        help="Use YOLOv8 detector instead of color-based")
    parser.add_argument("--yolo-model", default=None, metavar="PATH",
                        help="Path to YOLO .pt model file")
    args = parser.parse_args()

    use_usb = args.usb is not None

    # Override stream URL if provided
    if args.stream_url:
        config.ESP32_STREAM_URL = args.stream_url

    # Initialize detector
    if args.yolo or args.yolo_model:
        try:
            from yolo_detector import YOLODetector
            model_path = args.yolo_model or config.YOLO_MODEL_PATH
            detector = YOLODetector(model_path=model_path)
            det_mode = f"YOLO ({model_path})"
        except ImportError:
            print("[!] ultralytics not installed — falling back to color detection")
            print("    Install with: pip install ultralytics")
            detector = BallDetector()
            det_mode = "Color (YOLO unavailable)"
        except Exception as e:
            print(f"[!] YOLO init failed: {e} — falling back to color detection")
            detector = BallDetector()
            det_mode = "Color (YOLO failed)"
    else:
        detector = BallDetector()
        det_mode = "Color (HSV+YCrCb)"

    # Determine camera mode
    if use_usb:
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
    print(f"  Camera:    {cam_mode}")
    print("=" * 60)

    # Start grab thread
    if use_usb:
        # Open camera on MAIN thread (macOS requires this for permission dialog)
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

    # Start processing thread (runs detection on latest frame)
    proc_thread = threading.Thread(target=process_loop, daemon=True)
    proc_thread.start()

    # Give threads a moment to start
    time.sleep(1)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
