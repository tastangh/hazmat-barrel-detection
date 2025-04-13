import cv2
import os
import numpy as np

TEMPLATE_DIR = "./data/hazmats"

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def load_templates():
    templates = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0].replace("-", " ").title()
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = sift.detectAndCompute(img, None)
                if des is not None and len(des) >= 2:
                    templates.append((label, img, kp, des))
    print(f"[INFO] {len(templates)} hazmat şablonu yüklendi (SIFT).")
    return templates

TEMPLATES = load_templates()

def detect_hazmats(frame):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    if des_frame is None or len(des_frame) < 2:
        return []

    for label, tmpl, kp_tmpl, des_tmpl in TEMPLATES:
        if des_tmpl is None or len(des_tmpl) < 2:
            continue

        try:
            matches = flann.knnMatch(des_tmpl, des_frame, k=2)
        except cv2.error:
            print(f"[!] FLANN hata verdi: {label}")
            continue

        good_matches = []  
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 25:
            src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask is not None and np.sum(mask) > 15:
                h, w = tmpl.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                x, y, w, h = cv2.boundingRect(dst)

                # Küçük kutuları filtrele
                if w < 40 or h < 40:
                    print(f"[!] Çok küçük kutu atlandı: {label} ({w}x{h})")
                    continue

                detections.append((label, (x, y, w, h)))
                print(f"[+] Tespit: {label} (eşleşme={len(good_matches)}, güvenli eşleşme={np.sum(mask)})")
            else:
                print(f"[-] Homografi yetersiz veya maske başarısız: {label} (eşleşme={len(good_matches)})")
        else:
            print(f"[-] Eşleşme yetersiz: {label} ({len(good_matches)})")

    return detections
