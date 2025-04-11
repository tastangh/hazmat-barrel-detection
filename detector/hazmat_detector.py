import cv2
import os

TEMPLATE_DIR = "./data/hazmats"

def load_templates():
    templates = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0]
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, 0)
            if img is not None:
                templates.append((label, img))
    print(f"[DEBUG] {len(templates)} adet template yüklendi.")  # << BURASI YENİ
    return templates

TEMPLATES = load_templates()

def detect_hazmats(frame):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for label, template in TEMPLATES:
        for scale in [1.0, 0.8, 0.6]:
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            w, h = resized_template.shape[::-1]
            res = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            max_val = res.max()

            if max_val >= threshold:
                print(f"[DEBUG] TESPİT EDİLDİ: {label} - Skor: {max_val:.2f}")
                loc = zip(*((res >= threshold).nonzero()[::-1]))
                for pt in loc:
                    detections.append((label, (pt[0], pt[1], w, h)))
                    break
                break  # Birden fazla ölçek denemeye gerek yok
    return detections
