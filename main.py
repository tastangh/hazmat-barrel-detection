import cv2
import sys
import os
from detector.orb_hazmat_detector import detect_hazmats
from detector.barrel_detector import detect_barrels
from utils.draw_utils import draw_detections

DETECTED_OBJECTS = set()

def setup_logger(log_path="log.txt"):
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger()

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n[Frame {frame_idx}] -------------------------------")

        barrels = detect_barrels(frame)
        hazmats = detect_hazmats(frame) if frame_idx % 5 == 0 else []

        combined = hazmats + barrels
        new_detections = []

        for label, (x, y, w, h) in combined:
            if label not in DETECTED_OBJECTS:
                DETECTED_OBJECTS.add(label)
                print(f"[+] Tespit: {label} @ Frame {frame_idx}")
                new_detections.append((label, (x, y, w, h)))

        # Frame numarasını videoya yaz
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        frame = draw_detections(frame, combined)
        cv2.imshow("Tespit", frame)

        key = cv2.waitKey(0 if new_detections else 1)
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Video dosyası (örn: ./data/tusas-odev1.mp4)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    setup_logger("logs/output.log")

    main(args.video)
