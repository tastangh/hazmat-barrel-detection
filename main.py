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

def draw_detections(frame, detections):
    for label, (x, y, w, h) in detections:
        color = (0, 255, 0)
        if "Varil" in label:
            color = (255, 0, 0) if "Mavi" in label else (0, 0, 255) 
        elif "Hazmat" in label or any(char.isdigit() for char in label): 
             color = (0, 255, 255) 

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(y, label_size[1] + 10)
        cv2.rectangle(frame, (x, y_label - label_size[1] - 10),
                      (x + label_size[0], y_label + base_line - 10), color, cv2.FILLED)

        cv2.putText(frame, label, (x, y_label - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def non_max_suppression(boxes_scores_labels, overlapThresh):
    if len(boxes_scores_labels) == 0:
        return []

    boxes_scores_labels.sort(key=lambda x: x[2], reverse=True)

    labels = [item[0] for item in boxes_scores_labels]
    boxes = [item[1] for item in boxes_scores_labels]
    scores = [item[2] for item in boxes_scores_labels]

    np_boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
    pick_indices = [] 

    x1 = np_boxes[:, 0]
    y1 = np_boxes[:, 1]
    x2 = np_boxes[:, 2]
    y2 = np_boxes[:, 3]
    area = (x2 - x1) * (y2 - y1) 

    idxs = np.arange(len(scores))

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick_indices.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        overlap = intersection / union

        idxs_to_delete = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        idxs = np.delete(idxs, idxs_to_delete)

    final_detections = [(labels[i], boxes[i]) for i in pick_indices]
    return final_detections
    
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
