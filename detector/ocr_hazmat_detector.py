import cv2
from detector.ocr_hazmat_detector import detect_hazmats  # << DEĞİŞTİ!
from detector.barrel_detector import detect_barrels
from utils.draw_utils import draw_detections

DETECTED_OBJECTS = set()

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        barrels = detect_barrels(frame)
        hazmats = detect_hazmats(frame) if frame_idx % 5 == 0 else []

        combined = hazmats + barrels
        new_detections = []

        for label, (x, y, w, h) in combined:
            if label not in DETECTED_OBJECTS:
                DETECTED_OBJECTS.add(label)
                print(f"[+] Tespit: {label}")
                new_detections.append((label, (x, y, w, h)))

        frame = draw_detections(frame, combined)
        cv2.imshow("Tespit", frame)

        if new_detections:
            if cv2.waitKey(0) == ord('q'):
                break
        else:
            if cv2.waitKey(1) == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Video dosya yolu (örn: ./data/tusas-odev1.mp4)")
    args = parser.parse_args()
    main(args.video)
