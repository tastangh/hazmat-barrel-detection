import cv2
from utils.video_utils import get_video_frames
from utils.draw_utils import draw_detections
from detector.hazmat_detector import detect_hazmats
from detector.barrel_detector import detect_barrels

video_path = "./data/tusas-odev1.mp4"
frames = get_video_frames(video_path)

detected_labels = set()

for i, frame in enumerate(frames):
    hazmats = detect_hazmats(frame)
    barrels = detect_barrels(frame)
    all_detections = hazmats + barrels

    # Yeni tespit varsa duraklat
    new_detected = []
    for label, _ in all_detections:
        if label not in detected_labels:
            print(f"[FRAME {i}] Yeni Tespit: {label}")
            new_detected.append(label)
            detected_labels.add(label)

    if new_detected:
        frame = draw_detections(frame, all_detections)
        cv2.imshow("Tespitler", frame)
        cv2.waitKey(0)  # kullanıcı tuşuna kadar bekle
    else:
        frame = draw_detections(frame, all_detections)
        cv2.imshow("Tespitler", frame)
        cv2.waitKey(1)  # hızlı geç

cv2.destroyAllWindows()
