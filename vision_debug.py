import cv2
import numpy as np

# ============= ROI - ADJUST THESE FIRST =============
USE_ROI = False  # Set True once you have good ROI values
ROI_X = 100
ROI_Y = 150
ROI_WIDTH = 500
ROI_HEIGHT = 200

# ============= COLOR RANGES =============
# Start VERY loose, then tighten
GREEN_LOWER = (35, 50, 50)
GREEN_UPPER = (85, 255, 255)

PURPLE_LOWER = (100, 50, 50)
PURPLE_UPPER = (170, 255, 255)

# Detection params
MIN_AREA = 300
MAX_AREA = 100000

# State
frozen_frame = None
is_frozen = False


def process_frame(frame):
    """Process a single frame and return all debug info"""
    if USE_ROI:
        roi = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
        offset = (ROI_X, ROI_Y)
    else:
        roi = frame
        offset = (0, 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create masks
    green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    purple_mask = cv2.inRange(hsv, np.array(PURPLE_LOWER), np.array(PURPLE_UPPER))

    # Clean masks
    kernel = np.ones((7, 7), np.uint8)
    green_clean = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_clean = cv2.morphologyEx(green_clean, cv2.MORPH_CLOSE, kernel)
    purple_clean = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
    purple_clean = cv2.morphologyEx(purple_clean, cv2.MORPH_CLOSE, kernel)

    output = frame.copy()

    # Draw ROI if using it
    if USE_ROI:
        cv2.rectangle(output, (ROI_X, ROI_Y), (ROI_X+ROI_WIDTH, ROI_Y+ROI_HEIGHT), (255, 255, 0), 2)

    balls = []

    # Find green balls
    contours, _ = cv2.findContours(green_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w//2 + offset[0]
            cy = y + h//2 + offset[1]
            balls.append(('G', cx, cy, x+offset[0], y+offset[1], w, h, area))
            cv2.rectangle(output, (x+offset[0], y+offset[1]), (x+offset[0]+w, y+offset[1]+h), (0, 255, 0), 3)
            cv2.putText(output, f"G:{area}", (x+offset[0], y+offset[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Find purple balls
    contours, _ = cv2.findContours(purple_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w//2 + offset[0]
            cy = y + h//2 + offset[1]
            balls.append(('P', cx, cy, x+offset[0], y+offset[1], w, h, area))
            cv2.rectangle(output, (x+offset[0], y+offset[1]), (x+offset[0]+w, y+offset[1]+h), (255, 0, 255), 3)
            cv2.putText(output, f"P:{area}", (x+offset[0], y+offset[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Sort left to right
    balls.sort(key=lambda b: b[1])
    order = ''.join([b[0] for b in balls])

    cv2.putText(output, f"Found: {len(balls)} | Order: {order}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return output, green_mask, green_clean, purple_mask, purple_clean, balls, order


def mouse_hsv(event, x, y, flags, param):
    """Show HSV on click"""
    if event == cv2.EVENT_LBUTTONDOWN and param is not None:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
            h, s, v = hsv[y, x]
            print(f"CLICKED ({x}, {y}) -> H:{h} S:{s} V:{v}")


def main():
    global frozen_frame, is_frozen

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Main")

    print("="*60)
    print("BALL DETECTION DEBUG MODE")
    print("="*60)
    print("CONTROLS:")
    print("  SPACE - Freeze/unfreeze frame (analyze static image)")
    print("  CLICK - Show HSV value at that pixel")
    print("  S     - Save current frame to 'captured.png'")
    print("  Q     - Quit")
    print()
    print("WORKFLOW:")
    print("  1. Press SPACE to freeze when balls are visible")
    print("  2. Look at the masks - are the balls showing as white?")
    print("  3. Click on balls to get their HSV values")
    print("  4. Adjust GREEN_LOWER/UPPER and PURPLE_LOWER/UPPER")
    print("  5. Adjust MIN_AREA if balls are too small/big")
    print("="*60)

    while True:
        if not is_frozen:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
        else:
            current_frame = frozen_frame

        output, green_raw, green_clean, purple_raw, purple_clean, balls, order = process_frame(current_frame)

        if is_frozen:
            cv2.putText(output, "*** FROZEN - Press SPACE to unfreeze ***", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.setMouseCallback("Main", mouse_hsv, current_frame)

        # Stack masks for display
        green_display = cv2.cvtColor(green_clean, cv2.COLOR_GRAY2BGR)
        purple_display = cv2.cvtColor(purple_clean, cv2.COLOR_GRAY2BGR)

        # Add labels to masks
        cv2.putText(green_display, "GREEN MASK (cleaned)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(purple_display, "PURPLE MASK (cleaned)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow("Main", output)
        cv2.imshow("Green Mask", green_display)
        cv2.imshow("Purple Mask", purple_display)

        # Also show raw masks to compare
        cv2.imshow("Green RAW", green_raw)
        cv2.imshow("Purple RAW", purple_raw)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if is_frozen:
                is_frozen = False
                print(">>> UNFROZEN - live feed")
            else:
                is_frozen = True
                frozen_frame = current_frame.copy()
                print(f">>> FROZEN - Detected: {order} ({len(balls)} balls)")
                for b in balls:
                    print(f"    {b[0]} at x={b[1]}, area={b[7]}")
        elif key == ord('s'):
            cv2.imwrite("captured.png", current_frame)
            cv2.imwrite("captured_green_mask.png", green_clean)
            cv2.imwrite("captured_purple_mask.png", purple_clean)
            print(">>> Saved: captured.png, captured_green_mask.png, captured_purple_mask.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()