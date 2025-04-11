import cv2
import numpy as np

def detect_barrels(frame):
    detections = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk için iki farklı aralık (HSV)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    # Mavi renk aralığı (HSV)
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    for mask, label in [(red_mask, "Kırmızı Varil"), (blue_mask, "Mavi Varil")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # Gürültüleri engellemek için
                x, y, w, h = cv2.boundingRect(c)
                detections.append((label, (x, y, w, h)))
    return detections
