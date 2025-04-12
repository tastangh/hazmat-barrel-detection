import cv2
import os
import numpy as np

TEMPLATE_DIR = "./data/hazmats"
TEMPLATE_SCALES = [0.4, 0.5, 0.6, 0.7, 0.8]  # Daha küçük ölçeklerde deniyoruz

def load_templates():
    templates = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0].replace("-", " ").title()
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((label, img))
    print(f"[INFO] {len(templates)} hazmat şablonu yüklendi.")
    return templates

TEMPLATES = load_templates()

def detect_hazmats(frame):
    detections = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for label, template in TEMPLATES:
        found = False
        for scale in TEMPLATE_SCALES:
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            if resized_template.shape[0] >= gray_frame.shape[0] or resized_template.shape[1] >= gray_frame.shape[1]:
                continue  # Template frame'den büyükse geç

            res = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.75:
                x, y = max_loc
                h, w = resized_template.shape
                detections.append((label, (x, y, w, h)))
                print(f"[HAZMAT] {label} ({scale}x) tespit edildi. Skor: {max_val:.2f}")
                found = True
                break  # İlk başarılı eşleşmede dur

        if not found:
            print(f"[HAZMAT] {label} bulunamadı.")
    return detections
