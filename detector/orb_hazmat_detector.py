import cv2
import os
import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def non_max_suppression(boxes_scores_labels, overlapThresh):
    if len(boxes_scores_labels) == 0:
        return []

    boxes_scores_labels.sort(key=lambda x: x[2], reverse=True)

    labels = [item[0] for item in boxes_scores_labels]
    boxes = [item[1] for item in boxes_scores_labels]
    scores = [item[2] for item in boxes_scores_labels]

    np_boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
    pick_indices = [] 

    x1 = np_boxes[:, 0]
    y1 = np_boxes[:, 1]
    x2 = np_boxes[:, 2]
    y2 = np_boxes[:, 3]
    area = (x2 - x1) * (y2 - y1) 

    idxs = np.arange(len(scores))

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick_indices.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        overlap = intersection / union

        idxs_to_delete = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        idxs = np.delete(idxs, idxs_to_delete)

    final_detections = [(labels[i], boxes[i]) for i in pick_indices]
    return final_detections

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
