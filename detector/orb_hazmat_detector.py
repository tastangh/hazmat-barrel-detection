import cv2
import os
import numpy as np
from utils.iou_nms import non_max_suppression 

TEMPLATE_DIR = "./data/hazmats"

KNN_RATIO = 0.6
MIN_GOOD_MATCHES = 18
MIN_RANSAC_INLIERS = 10
MIN_BOX_SIZE = 40
MIN_CONFIDENCE = 15
MIN_ASPECT_RATIO = 0.7
MAX_ASPECT_RATIO = 1.4
NMS_THRESHOLD_HAZMAT = 0.25
DARK_FRAME_THRESHOLD = 50  

try:
    sift = cv2.SIFT_create()
except cv2.error as e:
    print("[HATA] SIFT kullanılamıyor. Lütfen 'opencv-contrib-python' paketinin kurulu olduğundan emin olun.")
    exit()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=30) 
flann = cv2.FlannBasedMatcher(index_params, search_params)

CLAHE_OBJ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def load_templates():
    templates = []
    print(f"[BİLGİ] Şablonlar yükleniyor: {os.path.abspath(TEMPLATE_DIR)}")
    if not os.path.isdir(TEMPLATE_DIR):
        print(f"[HATA] Şablon klasörü bulunamadı: {TEMPLATE_DIR}")
        return []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            label_base = os.path.splitext(filename)[0]
            label_parts = label_base.split('-')
            label = " ".join(part.replace("_", " ").title() for part in label_parts)
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            kp, des = sift.detectAndCompute(img, None)
            if des is None or len(des) < 2:
                continue
            templates.append((label, img, kp, des))
    print(f"[BİLGİ] {len(templates)} hazmat şablonu yüklendi (SIFT).")
    return templates

TEMPLATES = load_templates()

# --- Eşleşme Fonksiyonu ---
def match_template(template_tuple, kp_frame, des_frame):
    label, tmpl, kp_tmpl, des_tmpl = template_tuple

    try:
        matches = flann.knnMatch(des_tmpl, des_frame, k=2)
    except:
        return None

    good_matches = []
    for m, n in matches:
        if m.distance < KNN_RATIO * n.distance:
            good_matches.append(m)

    if len(good_matches) < MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None or mask is None or np.sum(mask) < MIN_RANSAC_INLIERS:
        return None

    h_tmpl, w_tmpl = tmpl.shape
    pts_tmpl = np.float32([[0, 0], [0, h_tmpl - 1], [w_tmpl - 1, h_tmpl - 1], [w_tmpl - 1, 0]]).reshape(-1, 1, 2)

    try:
        dst_scene = cv2.perspectiveTransform(pts_tmpl, M)
    except:
        return None

    x, y, w, h = cv2.boundingRect(dst_scene.astype(np.int32))
    confidence_score = float(np.sum(mask))

    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
        return None

    aspect_ratio = float(w) / h if h > 0 else 0
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return None

    if confidence_score < MIN_CONFIDENCE:
        return None

    return (label, (x, y, w, h), confidence_score)

def detect_hazmats(frame):
    if not TEMPLATES:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)
    if mean_brightness < DARK_FRAME_THRESHOLD:
        gray = CLAHE_OBJ.apply(gray)
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)

    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    if des_frame is None or len(des_frame) < MIN_GOOD_MATCHES:
        return []

    potential_detections = []
    for tmpl in TEMPLATES:
        result = match_template(tmpl, kp_frame, des_frame)
        if result is not None:
            potential_detections.append(result)

    if not potential_detections:
        return []

    final_hazmat_detections = non_max_suppression(potential_detections, NMS_THRESHOLD_HAZMAT)
    return final_hazmat_detections
