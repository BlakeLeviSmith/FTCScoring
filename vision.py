#%%
import cv2
import numpy as np
from collections import deque

# Temporal smoothing: track last N frames for stable detection
class StableDetector:
    def __init__(self, history_size=10, stability_threshold=0.6):
        self.history = deque(maxlen=history_size)
        self.stability_threshold = stability_threshold
        self.locked_motif = ""
        self.lock_frames = 0
        self.LOCK_DURATION = 15  # Frames to hold a motif before allowing change

    def update(self, current_motif):
        self.history.append(current_motif)

        if len(self.history) < 3:
            return current_motif

        # Count occurrences of each motif in history
        motif_counts = {}
        for m in self.history:
            motif_counts[m] = motif_counts.get(m, 0) + 1

        # Find most common motif
        most_common = max(motif_counts, key=motif_counts.get)
        frequency = motif_counts[most_common] / len(self.history)

        # If we have a locked motif, keep it unless new one is very stable
        if self.locked_motif and self.lock_frames > 0:
            self.lock_frames -= 1
            # Only unlock if new motif is dominant and different
            if frequency >= 0.8 and most_common != self.locked_motif:
                self.locked_motif = most_common
                self.lock_frames = self.LOCK_DURATION
            return self.locked_motif

        # Lock onto stable motif
        if frequency >= self.stability_threshold:
            self.locked_motif = most_common
            self.lock_frames = self.LOCK_DURATION
            return most_common

        # Return locked motif if we have one, else most common
        return self.locked_motif if self.locked_motif else most_common

# Global detector instance
stable_detector = StableDetector()

def show_hsv_values(event, x, y, _flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        ycrcb_frame = cv2.cvtColor(param, cv2.COLOR_BGR2YCrCb)
        print(f"HSV: {hsv_frame[y, x]} | YCrCb: {ycrcb_frame[y, x]} at ({x}, {y})")

def is_semicircle(contour, min_area=500):
    """
    Detect if a contour is a semi-circular shape.
    Returns (is_semicircle, score) where score indicates confidence.
    Uses relaxed criteria to detect more semi-circles.
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return False, 0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False, 0

    # Circularity: semi-circles typically have circularity ~0.5-0.8
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    x, y, w, h = cv2.boundingRect(contour)

    # Aspect ratio: semi-circles are typically wider than tall (or vice versa)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

    # Extent: ratio of contour area to bounding rect area
    # Semi-circles fill ~50-65% of their bounding rectangle
    extent = area / (w * h) if w * h > 0 else 0

    # Convex hull analysis
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Fit ellipse if enough points (needs at least 5)
    ellipse_match = 0
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        ellipse_aspect = max(ellipse[1]) / min(ellipse[1]) if min(ellipse[1]) > 0 else 0
        # Semi-circles when fit to ellipse have aspect ratio ~1.2-3.5 (relaxed)
        if 1.2 < ellipse_aspect < 3.5:
            ellipse_match = 1

    # Relaxed semi-circle criteria to detect more shapes:
    # - Circularity between 0.3 and 0.95 (wider range)
    # - Aspect ratio between 1.1 and 3.5 (more elongation allowed)
    # - Extent between 0.35 and 0.85 (wider fill range)
    # - Solidity > 0.75 (slightly more irregular shapes allowed)

    is_semi = (
        0.3 < circularity < 0.95 and
        1.1 < aspect_ratio < 3.5 and
        0.35 < extent < 0.85 and
        solidity > 0.75
    )

    # Calculate confidence score
    score = 0
    if is_semi:
        # Ideal semi-circle values
        circ_score = max(0, 1 - abs(circularity - 0.6) / 0.4)
        aspect_score = max(0, 1 - abs(aspect_ratio - 1.8) / 1.0)
        extent_score = max(0, 1 - abs(extent - 0.55) / 0.3)
        score = area * solidity * (circ_score + aspect_score + extent_score + ellipse_match) / 4

    return is_semi, score


def get_dominant_color(contour, mask_green, mask_purple):
    """
    Determine color by sampling multiple points in the contour region.
    Returns 'G', 'P', or None based on majority voting.
    """
    x, y, w, h = cv2.boundingRect(contour)

    # Create a mask for just this contour
    contour_mask = np.zeros(mask_green.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    # Count pixels that match each color within the contour
    green_pixels = cv2.countNonZero(cv2.bitwise_and(mask_green, contour_mask))
    purple_pixels = cv2.countNonZero(cv2.bitwise_and(mask_purple, contour_mask))

    total = green_pixels + purple_pixels
    if total == 0:
        return None

    # Require at least 30% dominance to classify
    if green_pixels > purple_pixels and green_pixels / total > 0.3:
        return 'G'
    elif purple_pixels > green_pixels and purple_pixels / total > 0.3:
        return 'P'

    return None


# Default green thresholds (can be tuned with green_tuner())
GREEN_HSV_LOWER = np.array([35, 40, 40])
GREEN_HSV_UPPER = np.array([85, 255, 255])
GREEN_YCRCB_LOWER = np.array([40, 0, 0])
GREEN_YCRCB_UPPER = np.array([220, 115, 135])


def runPipeline(image, use_stabilization=True):
    global stable_detector
    output = image.copy()

    # Convert to both color spaces
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Green detection using DUAL MASK (HSV AND YCrCb)
    # A pixel must pass BOTH filters to be considered green
    mask_green_hsv = cv2.inRange(img_hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    mask_green_ycrcb = cv2.inRange(img_ycrcb, GREEN_YCRCB_LOWER, GREEN_YCRCB_UPPER)
    mask_green_raw = cv2.bitwise_and(mask_green_hsv, mask_green_ycrcb)

    # HSV ranges for purple detection (kept as before)
    lower_purple, upper_purple = (134, 50, 80), (170, 255, 255)
    mask_purple_raw = cv2.inRange(img_hsv, lower_purple, upper_purple)

    # Process each color mask SEPARATELY with appropriate morphology
    kernel = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    # Green mask cleaning
    mask_green = cv2.morphologyEx(mask_green_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Purple mask cleaning
    mask_purple = cv2.morphologyEx(mask_purple_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Combine for contour detection
    mask_combined = cv2.bitwise_or(mask_green, mask_purple)

    # Additional cleanup on combined
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_large)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-filter by area
    MIN_CONTOUR_AREA = 600
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    detected_shapes = []
    green_count = 0
    purple_count = 0

    for contour in contours:
        is_semi, score = is_semicircle(contour, min_area=MIN_CONTOUR_AREA)

        if is_semi:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2

            # Use robust color detection (multiple pixel sampling)
            color = get_dominant_color(contour, mask_green, mask_purple)

            if color == 'G':
                green_count += 1
                detected_shapes.append((center_x, 'G', contour, x, y, score))
                cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
                cv2.putText(output, f'G{green_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif color == 'P':
                purple_count += 1
                detected_shapes.append((center_x, 'P', contour, x, y, score))
                cv2.drawContours(output, [contour], -1, (255, 0, 255), 3)
                cv2.putText(output, f'P{purple_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    detected_shapes.sort(key=lambda b: b[0])
    raw_motif = ''.join([b[1] for b in detected_shapes])

    # Apply temporal stabilization
    if use_stabilization:
        stable_motif = stable_detector.update(raw_motif)
    else:
        stable_motif = raw_motif

    total_count = len(stable_motif)

    # Show both raw and stable motif for debugging
    cv2.putText(output, f"Stable: {stable_motif} ({total_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(output, f"Raw: {raw_motif}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(output, f"G:{green_count} P:{purple_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return stable_motif, output, mask_clean, mask_green, mask_purple


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Color Output")

    last_printed = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stable_motif, output, mask_combined, mask_green, mask_purple = runPipeline(frame)

        # Only print when motif changes (reduces console spam)
        if stable_motif and stable_motif != last_printed:
            print(f"Motif: {stable_motif}")
            last_printed = stable_motif

        cv2.setMouseCallback("Color Output", show_hsv_values, frame)

        cv2.imshow("Color Output", output)
        cv2.imshow("Mask (B&W)", mask_combined)
        cv2.imshow("Green Mask", mask_green)
        cv2.imshow("Purple Mask", mask_purple)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# %%
# HSV Value Checker - Click on any pixel to see its HSV values
# Run this cell to fine-tune your color thresholds

def hsv_checker():
    cap = cv2.VideoCapture(0)

    hsv_values = []
    ycrcb_values = []

    def on_click(event, x, y, _flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame, hsv_frame, ycrcb_frame = param
            hsv = hsv_frame[y, x]
            ycrcb = ycrcb_frame[y, x]
            bgr = frame[y, x]
            print(f"\n{'='*40}")
            print(f"Position: ({x}, {y})")
            print(f"HSV: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
            print(f"YCrCb: Y={ycrcb[0]}, Cr={ycrcb[1]}, Cb={ycrcb[2]}")
            print(f"BGR: B={bgr[0]}, G={bgr[1]}, R={bgr[2]}")
            hsv_values.append(hsv)
            ycrcb_values.append(ycrcb)

            if len(hsv_values) > 1:
                h_vals = [v[0] for v in hsv_values]
                s_vals = [v[1] for v in hsv_values]
                v_vals = [v[2] for v in hsv_values]
                y_vals = [v[0] for v in ycrcb_values]
                cr_vals = [v[1] for v in ycrcb_values]
                cb_vals = [v[2] for v in ycrcb_values]
                print(f"\n--- Suggested HSV range from {len(hsv_values)} samples ---")
                print(f"Lower: ({min(h_vals)-5}, {max(0, min(s_vals)-20)}, {max(0, min(v_vals)-20)})")
                print(f"Upper: ({max(h_vals)+5}, {min(255, max(s_vals)+20)}, {min(255, max(v_vals)+20)})")
                print(f"\n--- Suggested YCrCb range from {len(ycrcb_values)} samples ---")
                print(f"Lower: ({max(0, min(y_vals)-20)}, {max(0, min(cr_vals)-10)}, {max(0, min(cb_vals)-10)})")
                print(f"Upper: ({min(255, max(y_vals)+20)}, {min(255, max(cr_vals)+10)}, {min(255, max(cb_vals)+10)})")

    cv2.namedWindow("HSV Checker")
    print("HSV/YCrCb Checker Started")
    print("- LEFT CLICK on colors to sample values")
    print("- Press 'c' to clear samples")
    print("- Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cv2.setMouseCallback("HSV Checker", on_click, (frame, hsv_frame, ycrcb_frame))

        # Draw instructions on frame
        cv2.putText(frame, "Click to sample HSV/YCrCb | 'c'=clear | 'q'=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(hsv_values)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("HSV Checker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            hsv_values.clear()
            ycrcb_values.clear()
            print("\nSamples cleared!")

    cap.release()
    cv2.destroyAllWindows()

    if hsv_values:
        print(f"\n{'='*40}")
        print("FINAL SUMMARY")
        print(f"{'='*40}")
        h_vals = [v[0] for v in hsv_values]
        s_vals = [v[1] for v in hsv_values]
        v_vals = [v[2] for v in hsv_values]
        y_vals = [v[0] for v in ycrcb_values]
        cr_vals = [v[1] for v in ycrcb_values]
        cb_vals = [v[2] for v in ycrcb_values]
        print(f"Collected {len(hsv_values)} samples")
        print(f"\nHSV ranges:")
        print(f"H range: {min(h_vals)} - {max(h_vals)}")
        print(f"S range: {min(s_vals)} - {max(s_vals)}")
        print(f"V range: {min(v_vals)} - {max(v_vals)}")
        print(f"\nSuggested HSV threshold:")
        print(f"lower = ({max(0, min(h_vals)-5)}, {max(0, min(s_vals)-20)}, {max(0, min(v_vals)-20)})")
        print(f"upper = ({min(179, max(h_vals)+5)}, {min(255, max(s_vals)+20)}, {min(255, max(v_vals)+20)})")
        print(f"\nYCrCb ranges:")
        print(f"Y range: {min(y_vals)} - {max(y_vals)}")
        print(f"Cr range: {min(cr_vals)} - {max(cr_vals)}")
        print(f"Cb range: {min(cb_vals)} - {max(cb_vals)}")
        print(f"\nSuggested YCrCb threshold:")
        print(f"lower = ({max(0, min(y_vals)-20)}, {max(0, min(cr_vals)-10)}, {max(0, min(cb_vals)-10)})")
        print(f"upper = ({min(255, max(y_vals)+20)}, {min(255, max(cr_vals)+10)}, {min(255, max(cb_vals)+10)})")

# Uncomment to run:
# hsv_checker()

# %%
# GREEN DUAL-MASK TUNER
# Tune both HSV and YCrCb thresholds with real-time visual feedback
import cv2
import numpy as np
from collections import deque
def green_tuner():
    """
    Interactive tuner for green ball detection using dual HSV + YCrCb masks.
    Shows individual masks and the combined AND result.
    Press 's' to save values, 'q' to quit.
    """
    cap = cv2.VideoCapture(0)

    # Create windows
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original + Detection")
    cv2.namedWindow("HSV Mask")
    cv2.namedWindow("YCrCb Mask")
    cv2.namedWindow("Combined (AND)")

    # HSV trackbars (H: 0-179, S: 0-255, V: 0-255)
    cv2.createTrackbar("H_min", "Controls", 35, 179, lambda x: None)
    cv2.createTrackbar("H_max", "Controls", 85, 179, lambda x: None)
    cv2.createTrackbar("S_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("S_max", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("V_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("V_max", "Controls", 255, 255, lambda x: None)

    # YCrCb trackbars (all 0-255)
    cv2.createTrackbar("Y_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("Y_max", "Controls", 220, 255, lambda x: None)
    cv2.createTrackbar("Cr_min", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("Cr_max", "Controls", 115, 255, lambda x: None)
    cv2.createTrackbar("Cb_min", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("Cb_max", "Controls", 135, 255, lambda x: None)

    # Morphology controls
    cv2.createTrackbar("Morph_kernel", "Controls", 5, 15, lambda x: None)
    cv2.createTrackbar("Open_iter", "Controls", 1, 5, lambda x: None)
    cv2.createTrackbar("Close_iter", "Controls", 2, 5, lambda x: None)

    # Step size for keyboard adjustments
    STEP_H = 2
    STEP_SV = 5
    STEP_YCRCB = 5

    print("="*60)
    print("GREEN DUAL-MASK TUNER")
    print("="*60)
    print("Use KEYBOARD to tune (sliders may not work on macOS)")
    print("Goal: Green ball = WHITE, everything else = BLACK")
    print("")
    print("KEYBOARD CONTROLS:")
    print("  HSV Hue:        1/! = H_min -/+     2/@ = H_max -/+")
    print("  HSV Sat:        3/# = S_min -/+     4/$ = S_max -/+")
    print("  HSV Val:        5/% = V_min -/+     6/^ = V_max -/+")
    print("  YCrCb Y:        7/& = Y_min -/+     8/* = Y_max -/+")
    print("  YCrCb Cr:       9/( = Cr_min -/+    0/) = Cr_max -/+")
    print("  YCrCb Cb:       -/_ = Cb_min -/+    =/+ = Cb_max -/+")
    print("")
    print("  s = Save values    r = Reset    q = Quit")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get trackbar values
        h_min = cv2.getTrackbarPos("H_min", "Controls")
        h_max = cv2.getTrackbarPos("H_max", "Controls")
        s_min = cv2.getTrackbarPos("S_min", "Controls")
        s_max = cv2.getTrackbarPos("S_max", "Controls")
        v_min = cv2.getTrackbarPos("V_min", "Controls")
        v_max = cv2.getTrackbarPos("V_max", "Controls")

        y_min = cv2.getTrackbarPos("Y_min", "Controls")
        y_max = cv2.getTrackbarPos("Y_max", "Controls")
        cr_min = cv2.getTrackbarPos("Cr_min", "Controls")
        cr_max = cv2.getTrackbarPos("Cr_max", "Controls")
        cb_min = cv2.getTrackbarPos("Cb_min", "Controls")
        cb_max = cv2.getTrackbarPos("Cb_max", "Controls")

        morph_k = max(1, cv2.getTrackbarPos("Morph_kernel", "Controls"))
        open_iter = max(1, cv2.getTrackbarPos("Open_iter", "Controls"))
        close_iter = max(1, cv2.getTrackbarPos("Close_iter", "Controls"))

        # Convert color spaces
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Create masks
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

        lower_ycrcb = np.array([y_min, cr_min, cb_min])
        upper_ycrcb = np.array([y_max, cr_max, cb_max])
        mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)

        # Combined mask (AND operation)
        mask_combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Apply morphology to combined
        kernel = np.ones((morph_k, morph_k), np.uint8)
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        # Find contours and draw on output
        output = frame.copy()
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        for i, contour in enumerate(contours):
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cv2.putText(output, f"A:{area}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count pixels for feedback
        hsv_pixels = cv2.countNonZero(mask_hsv)
        ycrcb_pixels = cv2.countNonZero(mask_ycrcb)
        combined_pixels = cv2.countNonZero(mask_clean)

        # Add info text - current values and pixel counts
        cv2.putText(output, f"HSV: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(output, f"YCrCb: Y[{y_min}-{y_max}] Cr[{cr_min}-{cr_max}] Cb[{cb_min}-{cb_max}]",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(output, f"Pixels: HSV={hsv_pixels} YCrCb={ycrcb_pixels} Combined={combined_pixels} | Contours: {len(contours)}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show windows
        cv2.imshow("Original + Detection", output)
        cv2.imshow("HSV Mask", mask_hsv)
        cv2.imshow("YCrCb Mask", mask_ycrcb)
        cv2.imshow("Combined (AND)", mask_clean)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n" + "="*60)
            print("SAVED VALUES - Copy these to vision.py:")
            print("="*60)
            print(f"GREEN_HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"GREEN_HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"GREEN_YCRCB_LOWER = np.array([{y_min}, {cr_min}, {cb_min}])")
            print(f"GREEN_YCRCB_UPPER = np.array([{y_max}, {cr_max}, {cb_max}])")
            print(f"# Morphology: kernel={morph_k}, open_iter={open_iter}, close_iter={close_iter}")
            print("="*60 + "\n")
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos("H_min", "Controls", 35)
            cv2.setTrackbarPos("H_max", "Controls", 85)
            cv2.setTrackbarPos("S_min", "Controls", 40)
            cv2.setTrackbarPos("S_max", "Controls", 255)
            cv2.setTrackbarPos("V_min", "Controls", 40)
            cv2.setTrackbarPos("V_max", "Controls", 255)
            cv2.setTrackbarPos("Y_min", "Controls", 40)
            cv2.setTrackbarPos("Y_max", "Controls", 220)
            cv2.setTrackbarPos("Cr_min", "Controls", 0)
            cv2.setTrackbarPos("Cr_max", "Controls", 115)
            cv2.setTrackbarPos("Cb_min", "Controls", 0)
            cv2.setTrackbarPos("Cb_max", "Controls", 135)
            print("Reset to defaults!")

        # Keyboard controls for tuning (number keys and shift+number)
        # H_min: 1/!
        elif key == ord('1'):
            cv2.setTrackbarPos("H_min", "Controls", max(0, h_min - STEP_H))
        elif key == ord('!'):
            cv2.setTrackbarPos("H_min", "Controls", min(179, h_min + STEP_H))
        # H_max: 2/@
        elif key == ord('2'):
            cv2.setTrackbarPos("H_max", "Controls", max(0, h_max - STEP_H))
        elif key == ord('@'):
            cv2.setTrackbarPos("H_max", "Controls", min(179, h_max + STEP_H))
        # S_min: 3/#
        elif key == ord('3'):
            cv2.setTrackbarPos("S_min", "Controls", max(0, s_min - STEP_SV))
        elif key == ord('#'):
            cv2.setTrackbarPos("S_min", "Controls", min(255, s_min + STEP_SV))
        # S_max: 4/$
        elif key == ord('4'):
            cv2.setTrackbarPos("S_max", "Controls", max(0, s_max - STEP_SV))
        elif key == ord('$'):
            cv2.setTrackbarPos("S_max", "Controls", min(255, s_max + STEP_SV))
        # V_min: 5/%
        elif key == ord('5'):
            cv2.setTrackbarPos("V_min", "Controls", max(0, v_min - STEP_SV))
        elif key == ord('%'):
            cv2.setTrackbarPos("V_min", "Controls", min(255, v_min + STEP_SV))
        # V_max: 6/^
        elif key == ord('6'):
            cv2.setTrackbarPos("V_max", "Controls", max(0, v_max - STEP_SV))
        elif key == ord('^'):
            cv2.setTrackbarPos("V_max", "Controls", min(255, v_max + STEP_SV))
        # Y_min: 7/&
        elif key == ord('7'):
            cv2.setTrackbarPos("Y_min", "Controls", max(0, y_min - STEP_YCRCB))
        elif key == ord('&'):
            cv2.setTrackbarPos("Y_min", "Controls", min(255, y_min + STEP_YCRCB))
        # Y_max: 8/*
        elif key == ord('8'):
            cv2.setTrackbarPos("Y_max", "Controls", max(0, y_max - STEP_YCRCB))
        elif key == ord('*'):
            cv2.setTrackbarPos("Y_max", "Controls", min(255, y_max + STEP_YCRCB))
        # Cr_min: 9/(
        elif key == ord('9'):
            cv2.setTrackbarPos("Cr_min", "Controls", max(0, cr_min - STEP_YCRCB))
        elif key == ord('('):
            cv2.setTrackbarPos("Cr_min", "Controls", min(255, cr_min + STEP_YCRCB))
        # Cr_max: 0/)
        elif key == ord('0'):
            cv2.setTrackbarPos("Cr_max", "Controls", max(0, cr_max - STEP_YCRCB))
        elif key == ord(')'):
            cv2.setTrackbarPos("Cr_max", "Controls", min(255, cr_max + STEP_YCRCB))
        # Cb_min: -/_
        elif key == ord('-'):
            cv2.setTrackbarPos("Cb_min", "Controls", max(0, cb_min - STEP_YCRCB))
        elif key == ord('_'):
            cv2.setTrackbarPos("Cb_min", "Controls", min(255, cb_min + STEP_YCRCB))
        # Cb_max: =/+
        elif key == ord('='):
            cv2.setTrackbarPos("Cb_max", "Controls", max(0, cb_max - STEP_YCRCB))
        elif key == ord('+'):
            cv2.setTrackbarPos("Cb_max", "Controls", min(255, cb_max + STEP_YCRCB))

    cap.release()
    cv2.destroyAllWindows()


# Run the tuner:
green_tuner()