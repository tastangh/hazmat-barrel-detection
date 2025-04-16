import cv2
import sys
import os
import argparse
import numpy as np 
from detector.orb_hazmat_detector import detect_hazmats
from detector.barrel_detector import detect_barrels
from utils.draw_utils import draw_detections
from utils.iou_nms import calculate_iou

DETECTED_OBJECTS = set()

HAZMAT_DETECTION_INTERVAL = 3  # Her N frame'de bir hazmat tespiti yapma
IOU_THRESHOLD_BARREL_HAZMAT = 0.2  # Bir varil ile hazmat çakışıyorsa (IoU), varil iptal etme
PAUSE_ON_NEW_DETECTION = True  # Yeni bir nesne türü tespit edilirse video duraklatma

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
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    original_stdout = sys.stdout
    logger = Logger()
    sys.stdout = logger
    return logger, original_stdout

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video dosyası açılamadı: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[BİLGİ] Video Özellikleri: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    frame_idx = 0
    hazmats = []  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n[BİLGİ] Video akışı sona erdi.")
            break

        barrels = detect_barrels(frame)  # Varil tespiti her frame'de 

        if frame_idx % HAZMAT_DETECTION_INTERVAL == 0:
            hazmats = detect_hazmats(frame)  # Hazmatlar belirli aralıklarla tespit 

        final_barrels = []
        hazmat_boxes = [h[1] for h in hazmats]
        for barrel_label, barrel_box in barrels:
            is_overlapping_with_hazmat = False
            for hazmat_box in hazmat_boxes:
                iou = calculate_iou(barrel_box, hazmat_box)
                if iou > IOU_THRESHOLD_BARREL_HAZMAT:
                    is_overlapping_with_hazmat = True
                    break  # Bu varil bir hazmat ile çakışıyorsa kontrolü 
            if not is_overlapping_with_hazmat:
                final_barrels.append((barrel_label, barrel_box))

        combined_detections = hazmats + final_barrels  

        new_detections_this_frame = combined_detections
        for label, box in combined_detections:
            print(f"[*] >>>  Nesne Türü Tespit Edildi: {normalized_label} @ Frame {frame_idx}")

        # Frame üzerine kaçıncı frame olduğu bilgisi 
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_display = draw_detections(frame.copy(), combined_detections) 
        cv2.imshow("Detections", frame_display)

        # Kullanıcı kontrolü
        wait_time = 0 if (PAUSE_ON_NEW_DETECTION and new_detections_this_frame) else 1
        key = cv2.waitKey(wait_time) & 0xFF 
        if key == ord('q'):
            print("\n[BİLGİ] 'q' tuşuna basıldı. Çıkılıyor.")
            break
        elif key == ord('p'): 
             print("[BİLGİ] Duraklatıldı. Devam etmek için bir tuşa (q hariç) basın.")
             while True:
                 key_resume = cv2.waitKey(0) & 0xFF
                 if key_resume == ord('q'):
                     key = ord('q') 
                     break
                 if key_resume is not None:
                     print("[BİLGİ] Devam ediliyor.")
                     break
             if key == ord('q'):
                 break
        elif key == ord(' '): 
            pass 

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[BİLGİ] Kaynaklar serbest bırakıldı.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bir videoda Hazmat levhalarını ve varilleri tespit eder.")
    parser.add_argument("--video", type=str, required=True, help="Giriş video dosyasının yolu (örn: ./data/tusas-odev1.mp4)")
    parser.add_argument("--log", type=str, default="logs/output.log", help="Log dosyasının yolu.")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[HATA] Video dosyası bulunamadı: {args.video}")
        sys.exit(1)

    logger, original_stdout = setup_logger(args.log)
    print("--- Tespit Scripti Başlatıldı ---")
    print(f"Video Girdisi: {args.video}")
    print(f"Log Dosyası: {args.log}")
    print(f"Hazmat Tespit Aralığı: {HAZMAT_DETECTION_INTERVAL} frame")
    print(f"Varil/Hazmat Çakışma Eşiği (IoU): {IOU_THRESHOLD_BARREL_HAZMAT}")
    print(f"Yeni Tespitte Duraklat: {PAUSE_ON_NEW_DETECTION}")
    print("--------------------------------")

    try:
        main(args.video)
    except Exception as e:
        print(f"\n[FATAL HATA] Beklenmeyen bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        if hasattr(logger, 'close'):
            logger.close()
        print("--- Tespit Scripti Sonlandırıldı ---")