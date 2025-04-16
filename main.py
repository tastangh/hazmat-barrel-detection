import cv2
import sys
import os
import argparse
import numpy as np 
from detector.orb_hazmat_detector import detect_hazmats
from detector.barrel_detector import detect_barrels
from utils.draw_utils import draw_detections
from utils.iou_nms import calculate_iou

HAZMAT_DETECTION_INTERVAL = 3  # Her N frame'de bir hazmat tespiti yapma
IOU_THRESHOLD_BARREL_HAZMAT = 0.2  # Bir varil ile hazmat çakışıyorsa (IoU), varil iptal etme
PAUSE_ON_NEW_DETECTION = True  # Her tespitte video durdurulacak

def setup_logger(log_path="log.txt"):
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path, "a", encoding="utf-8", buffering=1)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger()

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video dosyası açılamadı.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hazmat_detections = []
        if frame_count % HAZMAT_DETECTION_INTERVAL == 0:
            hazmat_detections = detect_hazmats(frame)

        barrel_detections = detect_barrels(frame)

        # Çakışan varilleri çıkar
        filtered_barrels = []
        for barrel in barrel_detections:
            overlap = False
            for hazmat in hazmat_detections:
                iou = calculate_iou(barrel["bbox"], hazmat["bbox"])
                if iou > IOU_THRESHOLD_BARREL_HAZMAT:
                    overlap = True
                    break
            if not overlap:
                filtered_barrels.append(barrel)

        all_detections = hazmat_detections + filtered_barrels

        # Tüm tespitler için duraklatma ve terminal bildirimi
        for detection in all_detections:
            label = detection["label"]
            print(f"[FRAME {frame_count}] [DETECTED] Object: {label}")
            if PAUSE_ON_NEW_DETECTION:
                print("[INFO] Object detected. Press any key to continue...")
                cv2.imshow("Frame", frame)
                cv2.waitKey(0)

        draw_detections(frame, all_detections)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/test.mp4", help="Video dosya yolu")
    args = parser.parse_args()

    setup_logger()
    main(args.video)
