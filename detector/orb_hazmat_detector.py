import cv2
import os
import numpy as np

TEMPLATE_DIR = "./data/hazmats"

orb = cv2.ORB_create(nfeatures=1500)

FLANN_INDEX_LSH = 6
flann_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
flann = cv2.FlannBasedMatcher(flann_params, {})

def load_templates():
    templates = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0].replace("-", " ").title()
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                if des is not None:
                    templates.append((label, img, kp, des))
    print(f"[INFO] {len(templates)} hazmat şablonu yüklendi (ORB).")
    return templates

TEMPLATES = load_templates()

def detect_hazmats(frame):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is None:
        return []

    for label, template_img, kp_temp, des_temp in TEMPLATES:
        if des_temp is None:
            continue

        try:
            matches = flann.knnMatch(des_temp, des_frame, k=2)
        except cv2.error:
            continue

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.8 * n.distance:  # daha sıkı eşik
                    good_matches.append(m)

        if len(good_matches) > 250:  # daha yüksek eşik
            src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])
            x, y, w, h = cv2.boundingRect(dst_pts.reshape(-1, 1, 2))
            detections.append((label, (x, y, w, h)))
            print(f"[ORB] {label} TESPİT (match: {len(good_matches)})")
        else:
            print(f"[ORB] {label} EŞLEŞME YETERSİZ ({len(good_matches)})")

    return detections
