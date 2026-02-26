import cv2
import numpy as np

# ============= ROI CONFIGURATION =============
# Set these values to crop to just the ramp area
# Run with CALIBRATE_ROI = True first to find the values
CALIBRATE_ROI = True  # Set to True to draw ROI on screen
ROI_X = 200      # Left edge of ramp
ROI_Y = 200      # Top edge of ramp
ROI_WIDTH = 800  # Width of ramp area
ROI_HEIGHT = 300 # Height of ramp area

# ============= COLOR CONFIGURATION =============
# These will be calibrated - start with tighter ranges
# After running calibration, update these values
COLORS = {
    'green': {
        'lower': (40, 100, 100),   # Tighter saturation/value
        'upper': (80, 255, 255),
        'display_color': (0, 255, 0),
        'label': 'G'
    },
    'purple': {
        'lower': (120, 80, 80),    # Tighter saturation/value
        'upper': (160, 255, 255),
        'display_color': (255, 0, 255),
        'label': 'P'
    }
}

# ============= DETECTION PARAMETERS =============
MIN_BALL_AREA = 800       # Minimum contour area
MAX_BALL_AREA = 50000     # Maximum contour area
MIN_CIRCULARITY = 0.5     # How round the shape must be
MIN_SOLIDITY = 0.7        # How filled in the shape is

# Global for color sampling
sampled_colors = []

def mouse_callback(event, x, y, flags, param):
    """Click to sample HSV values at that point"""
    frame, roi_offset = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust for ROI offset if using ROI
        actual_x = x + roi_offset[0] if not CALIBRATE_ROI else x
        actual_y = y + roi_offset[1] if not CALIBRATE_ROI else y

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_value = hsv_frame[actual_y, actual_x]
        print(f"CLICKED - HSV at ({actual_x}, {actual_y}): H={hsv_value[0]}, S={hsv_value[1]}, V={hsv_value[2]}")
        sampled_colors.append(hsv_value)

        if len(sampled_colors) >= 5:
            avg_h = int(np.mean([c[0] for c in sampled_colors[-5:]]))
            avg_s = int(np.mean([c[1] for c in sampled_colors[-5:]]))
            avg_v = int(np.mean([c[2] for c in sampled_colors[-5:]]))
            print(f"  --> Last 5 samples average: H={avg_h}, S={avg_s}, V={avg_v}")
            print(f"  --> Suggested range: ({avg_h-10}, {avg_s-50}, {avg_v-50}) to ({avg_h+10}, 255, 255)")

    elif event == cv2.EVENT_MOUSEMOVE:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        actual_x = x + roi_offset[0] if not CALIBRATE_ROI else x
        actual_y = y + roi_offset[1] if not CALIBRATE_ROI else y
        if 0 <= actual_y < hsv_frame.shape[0] and 0 <= actual_x < hsv_frame.shape[1]:
            hsv_value = hsv_frame[actual_y, actual_x]
            # Update window title with HSV (less spammy than printing)


def detect_balls_of_color(hsv_roi, color_config, roi_offset):
    """Detect all balls of a specific color"""
    lower = np.array(color_config['lower'])
    upper = np.array(color_config['upper'])

    mask = cv2.inRange(hsv_roi, lower, upper)

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < MIN_CIRCULARITY:
            continue

        # Check solidity (convex hull fill)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue

        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2 + roi_offset[0]  # Add ROI offset back
        center_y = y + h // 2 + roi_offset[1]

        balls.append({
            'x': x + roi_offset[0],
            'y': y + roi_offset[1],
            'w': w,
            'h': h,
            'center_x': center_x,
            'center_y': center_y,
            'area': area,
            'label': color_config['label'],
            'color': color_config['display_color']
        })

    return balls, mask


def run_detection(frame):
    """Main detection pipeline"""
    output = frame.copy()

    # Apply ROI if not calibrating
    if CALIBRATE_ROI:
        # Draw ROI rectangle for calibration
        cv2.rectangle(output, (ROI_X, ROI_Y),
                     (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT),
                     (0, 255, 255), 2)
        cv2.putText(output, "CALIBRATION MODE - Adjust ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output, "Click on balls to sample HSV values",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        roi = frame
        roi_offset = (0, 0)
    else:
        # Crop to ROI
        roi = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
        roi_offset = (ROI_X, ROI_Y)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    all_balls = []
    all_masks = {}

    # Detect each color
    for color_name, color_config in COLORS.items():
        balls, mask = detect_balls_of_color(hsv_roi, color_config, roi_offset)
        all_balls.extend(balls)
        all_masks[color_name] = mask

    # Sort by x position (left to right)
    all_balls.sort(key=lambda b: b['center_x'])

    # Draw detections
    for ball in all_balls:
        cv2.rectangle(output, (ball['x'], ball['y']),
                     (ball['x'] + ball['w'], ball['y'] + ball['h']),
                     ball['color'], 2)
        cv2.putText(output, ball['label'],
                   (ball['x'], ball['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball['color'], 2)

    # Build order string
    order = ''.join([b['label'] for b in all_balls])

    cv2.putText(output, f"Detected: {len(all_balls)} balls",
               (10, output.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Order: {order}",
               (10, output.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return order, output, all_masks


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Detection")

    last_order = ""
    stable_count = 0
    STABLE_THRESHOLD = 10  # Need same result for 10 frames before printing

    print("=" * 50)
    print("BALL DETECTION - CALIBRATION MODE")
    print("=" * 50)
    print("1. Adjust ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT to frame just the ramp")
    print("2. Click on GREEN balls to sample their HSV values")
    print("3. Click on PURPLE balls to sample their HSV values")
    print("4. Update COLORS dict with the suggested ranges")
    print("5. Set CALIBRATE_ROI = False")
    print("=" * 50)
    print("Press 'q' to quit, 'r' to reset samples")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        order, output, masks = run_detection(frame)

        # Only print when order stabilizes and changes
        if order == last_order:
            stable_count += 1
        else:
            stable_count = 0
            last_order = order

        if stable_count == STABLE_THRESHOLD and order:
            print(f">>> DETECTED ORDER: {order} ({len(order)} balls)")
            stable_count += 1  # Prevent re-printing

        # Set up mouse callback for color sampling
        roi_offset = (0, 0) if CALIBRATE_ROI else (ROI_X, ROI_Y)
        cv2.setMouseCallback("Detection", mouse_callback, (frame, roi_offset))

        cv2.imshow("Detection", output)

        # Show individual masks
        for color_name, mask in masks.items():
            cv2.imshow(f"{color_name.capitalize()} Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sampled_colors.clear()
            print("Samples cleared!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()