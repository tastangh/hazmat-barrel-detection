import cv2
import numpy as np

MIN_BARREL_AREA = 800
BARREL_MIN_ASPECT_RATIO = 0.3 
BARREL_MAX_ASPECT_RATIO = 0.8

def detect_barrels(frame):
    detections = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 130, 80])
    red_upper1 = np.array([8, 255, 255])
    red_lower2 = np.array([172, 130, 80])
    red_upper2 = np.array([179, 255, 255])

    blue_lower = np.array([98, 160, 60])
    blue_upper = np.array([130, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)


    for mask, label in [(red_mask, "Kirmizi Varil"), (blue_mask, "Mavi Varil")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area > MIN_BARREL_AREA:
                x, y, w, h = cv2.boundingRect(c)

                if h > 0:
                    aspect_ratio = float(w) / h
                    if BARREL_MIN_ASPECT_RATIO <= aspect_ratio <= BARREL_MAX_ASPECT_RATIO:
                        detections.append((label, (x, y, w, h)))

    return detections
