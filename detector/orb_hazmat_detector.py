import cv2
import os

TEMPLATE_DIR = "./data/hazmats"
orb = cv2.ORB_create(nfeatures=1000)  # Daha fazla keypoint için

def load_templates():
    templates = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0]
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, 0)
            if img is not None:
                keypoints, descriptors = orb.detectAndCompute(img, None)
                templates.append((label, img, keypoints, descriptors))
    print(f"[DEBUG] {len(templates)} adet ORB template yüklendi.")
    return templates

TEMPLATES = load_templates()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def detect_hazmats(frame):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_scene, descriptors_scene = orb.detectAndCompute(gray, None)

    if descriptors_scene is None:
        return []

    for label, template_img, keypoints_template, descriptors_template in TEMPLATES:
        if descriptors_template is None:
            continue

        matches = bf.match(descriptors_template, descriptors_scene)
        matches = sorted(matches, key=lambda x: x.distance)

        # Eşik: iyi eşleşme sayısı
        good_matches = [m for m in matches if m.distance < 60]

        print(f"[DEBUG] {label}: {len(good_matches)} eşleşme")

        if len(good_matches) > 10:
            # Tespit edildi kabul et, rectangle çiz (şu anlık tüm sahneye yayalım)
            x, y, w, h = 50, 50, 200, 200  # Geçici sabit konum
            detections.append((label, (x, y, w, h)))

    return detections
