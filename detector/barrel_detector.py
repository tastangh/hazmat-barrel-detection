# --- START OF FILE barrel_detector.py ---

import cv2
import numpy as np

# --- Tunable Parameters ---
MIN_BARREL_AREA = 600 # Increased slightly
# --- End Tunable Parameters ---

def detect_barrels(frame):
    """Detects red and blue barrels based on color thresholding."""
    detections = [] # List of (label, (x, y, w, h))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges
    # Red (can wrap around 0/180)
    red_lower1 = np.array([0, 120, 70])   # Adjusted ranges might be needed
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70]) # Adjusted ranges might be needed
    red_upper2 = np.array([179, 255, 255])

    # Blue
    blue_lower = np.array([100, 150, 50]) # Adjusted ranges might be needed
    blue_upper = np.array([140, 255, 255])

    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Optional: Morphological operations to reduce noise and fill gaps
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
            if area > MIN_BARREL_AREA:  # Filter small contours
                x, y, w, h = cv2.boundingRect(c)
                # Optional: Add aspect ratio check for barrels if needed
                # aspect_ratio = float(w) / h
                # if 0.3 < aspect_ratio < 0.8: # Example: barrels are typically taller than wide
                detections.append((label, (x, y, w, h)))

    return detections

# --- END OF FILE barrel_detector.py ---