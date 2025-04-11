import cv2
from detector.orb_hazmat_detector import detect_hazmats
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

        # 1️⃣ Varil tespiti her karede çalışsın (hafif)
        barrels = detect_barrels(frame)

        # 2️⃣ Hazmat tespiti sadece her 5. karede çalışsın (ağır)
        if frame_idx % 5 == 0:
            hazmats = detect_hazmats(frame)
        else:
            hazmats = []

        combined = hazmats + barrels
        new_detections = []

        for label, (x, y, w, h) in combined:
            if label not in DETECTED_OBJECTS:
                DETECTED_OBJECTS.add(label)
                print(f"[+] Tespit: {label}")
                new_detections.append((label, (x, y, w, h)))

        # 3️⃣ Tüm tespitleri çiz
        frame = draw_detections(frame, combined)
        cv2.imshow("Tespit", frame)

        # 4️⃣ Duraksama: sadece yeni tespit varsa
        if new_detections:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        else:
            key = cv2.waitKey(1)  # ⚡️ 1ms bekleme ile hızlı oynatma
            if key == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Video dosya yolu (örn: ./data/odev1-tusas.mp4)")
    args = parser.parse_args()
    main(args.video)
