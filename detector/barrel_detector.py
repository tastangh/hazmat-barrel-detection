# --- START OF FILE barrel_detector.py ---

import cv2
import numpy as np

# --- Tunable Parameters ---
MIN_BARREL_AREA = 800 # INCREASED: Require larger barrels (was 600)
BARREL_MIN_ASPECT_RATIO = 0.3 # ADDED: Width/Height ratio (barrels are taller than wide)
BARREL_MAX_ASPECT_RATIO = 0.8 # ADDED: Width/Height ratio
# --- End Tunable Parameters ---

def detect_barrels(frame):
    """Detects red and blue barrels based on color thresholding and aspect ratio."""
    detections = [] # List of (label, (x, y, w, h))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- IMPORTANT: TUNE THESE HSV RANGES FOR YOUR VIDEO ---
    # Use a color picker tool on sample frames of your specific barrels!
    # Example Ranges (likely need adjustment):
    # Red (might need tuning based on lighting)
    red_lower1 = np.array([0, 130, 80])
    red_upper1 = np.array([8, 255, 255])
    red_lower2 = np.array([172, 130, 80])
    red_upper2 = np.array([179, 255, 255])

    # Blue (might need tuning based on lighting)
    blue_lower = np.array([98, 160, 60])
    blue_upper = np.array([130, 255, 255])
    # --- END OF HSV RANGES TO TUNE ---


    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Optional: Morphological operations (adjust kernel size if needed)
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)


    # Find contours for each color
    for mask, label in [(red_mask, "Kırmızı Varil"), (blue_mask, "Mavi Varil")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            # Check Area (stricter MIN_BARREL_AREA)
            if area > MIN_BARREL_AREA:
                x, y, w, h = cv2.boundingRect(c)

                # Check Aspect Ratio (NEW)
                if h > 0: # Avoid division by zero
                    aspect_ratio = float(w) / h
                    if BARREL_MIN_ASPECT_RATIO <= aspect_ratio <= BARREL_MAX_ASPECT_RATIO:
                        detections.append((label, (x, y, w, h)))
                    # else:
                        # print(f"[DEBUG] Barrel Rejected (Aspect Ratio): {label} ({aspect_ratio:.2f})")
                # else: Barrel has no height? reject.

    return detections

# --- END OF FILE barrel_detector.py ---