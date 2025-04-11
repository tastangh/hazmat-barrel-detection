import cv2

def draw_detections(frame, detections):
    for label, (x, y, w, h) in detections:
        color = (0, 255, 0) if "Varil" in label else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
