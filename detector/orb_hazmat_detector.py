# --- START OF FILE orb_hazmat_detector.py ---

import cv2
import os
import numpy as np
from utils.iou_nms import non_max_suppression # Import NMS

TEMPLATE_DIR = "./data/hazmats"

# --- Tunable Parameters --- Stricter Values ---
KNN_RATIO = 0.6              # LOWERED: Stricter initial match (was 0.65)
MIN_GOOD_MATCHES = 18        # INCREASED: Need more potential matches (was 15)
MIN_RANSAC_INLIERS = 10      # INCREASED: Need more geometrically consistent matches (was 8)
MIN_BOX_SIZE = 40            # INCREASED slightly (was 35)
MIN_CONFIDENCE = 15          # INCREASED: Final confidence check (was 12) - Often set >= MIN_RANSAC_INLIERS
MIN_ASPECT_RATIO = 0.7       # TIGHTENED Range (was 0.6)
MAX_ASPECT_RATIO = 1.4       # TIGHTENED Range (was 1.6) - Placards are squarish
NMS_THRESHOLD_HAZMAT = 0.25  # DECREASED slightly: More aggressive NMS (was 0.3)
# --- End Tunable Parameters ---

# Initialize SIFT and FLANN
try:
    sift = cv2.SIFT_create()
except cv2.error as e:
    print("[ERROR] SIFT not available. Ensure you have 'opencv-contrib-python' installed.")
    print("Try: pip uninstall opencv-python opencv-contrib-python")
    print("Then: pip install opencv-contrib-python")
    exit()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # Increase checks for potentially better accuracy but slower speed
flann = cv2.FlannBasedMatcher(index_params, search_params)

# --- Load Templates ---
def load_templates():
    # ... (Keep the load_templates function exactly as it was in the previous version) ...
    templates = []
    print(f"[INFO] Loading templates from: {os.path.abspath(TEMPLATE_DIR)}")
    if not os.path.isdir(TEMPLATE_DIR):
        print(f"[ERROR] Template directory not found: {TEMPLATE_DIR}")
        return []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            label_base = os.path.splitext(filename)[0]
            label_parts = label_base.split('-')
            label = " ".join(part.replace("_", " ").title() for part in label_parts)
            path = os.path.join(TEMPLATE_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Could not load template image: {filename}")
                continue
            kp, des = sift.detectAndCompute(img, None)
            if des is None or len(des) < 2:
                print(f"[WARNING] No/few descriptors found for template: {label} ({filename})")
                continue
            templates.append({"label": label, "img": img, "kp": kp, "des": des})
    print(f"[INFO] {len(templates)} hazmat templates loaded (SIFT).")
    return templates

TEMPLATES = load_templates()
# --- End Load Templates ---


def detect_hazmats(frame):
    potential_detections = [] # Store potential detections before NMS: (label, box, score)
    if not TEMPLATES:
        # print("[WARNING] No hazmat templates loaded, skipping detection.") # Less verbose
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        kp_frame, des_frame = sift.detectAndCompute(gray, None)
    except cv2.error as e:
        # print(f"[ERROR] SIFT detection failed on frame: {e}") # Less verbose
        return []

    if des_frame is None or len(des_frame) < MIN_GOOD_MATCHES: # Check against stricter MIN_GOOD_MATCHES
        return []

    for template_data in TEMPLATES:
        label = template_data["label"]
        tmpl = template_data["img"]
        kp_tmpl = template_data["kp"]
        des_tmpl = template_data["des"]

        if des_tmpl is None or len(des_tmpl) < 2:
            continue

        try:
            matches = flann.knnMatch(des_tmpl, des_frame, k=2)
            good_matches = []
            # Apply Lowe's ratio test with stricter KNN_RATIO
            for m, n in matches:
                # Ensure indices are valid before accessing descriptor
                if m.queryIdx < len(des_tmpl) and m.trainIdx < len(des_frame) and \
                   n.queryIdx < len(des_tmpl) and n.trainIdx < len(des_frame):
                     if m.distance < KNN_RATIO * n.distance:
                         good_matches.append(m)
                else:
                     # This case should ideally not happen with correct knnMatch usage, but safety check
                     # print(f"[WARN] Invalid match index for {label}")
                     pass

        except cv2.error as e:
            # print(f"[!] FLANN matching error for '{label}': {e}. Skipping template.")
            continue

        # Check if enough good matches were found (stricter MIN_GOOD_MATCHES)
        if len(good_matches) >= MIN_GOOD_MATCHES:
            src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None and mask is not None:
                inliers = np.sum(mask)
                # Check against stricter MIN_RANSAC_INLIERS
                if inliers >= MIN_RANSAC_INLIERS:
                    h_tmpl, w_tmpl = tmpl.shape
                    pts_tmpl = np.float32([[0, 0], [0, h_tmpl - 1], [w_tmpl - 1, h_tmpl - 1], [w_tmpl - 1, 0]]).reshape(-1, 1, 2)
                    try:
                        dst_scene = cv2.perspectiveTransform(pts_tmpl, M)
                    except cv2.error:
                        continue

                    x, y, w, h = cv2.boundingRect(dst_scene.astype(np.int32))

                    # --- Filtering (with stricter values) ---
                    confidence_score = float(inliers)

                    # 1. Check minimum size (stricter MIN_BOX_SIZE)
                    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                        # print(f"[DEBUG] Rejected (Too Small): {label} ({w}x{h})")
                        continue

                    # 2. Check aspect ratio (stricter MIN/MAX_ASPECT_RATIO)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                        # print(f"[DEBUG] Rejected (Aspect Ratio): {label} ({aspect_ratio:.2f})")
                        continue

                    # 3. Check minimum confidence (stricter MIN_CONFIDENCE)
                    if confidence_score < MIN_CONFIDENCE:
                        # print(f"[DEBUG] Rejected (Low Confidence): {label} (Score: {confidence_score})")
                        continue

                    # --- Passed Filters ---
                    box = (x, y, w, h)
                    potential_detections.append((label, box, confidence_score))
                    # print(f"[DEBUG] Potential Add: {label} (Score: {confidence_score:.0f})")


    # Apply Non-Maximum Suppression (stricter NMS_THRESHOLD_HAZMAT)
    if not potential_detections:
        return []

    final_hazmat_detections = non_max_suppression(potential_detections, NMS_THRESHOLD_HAZMAT)
    # print(f"[DEBUG] Hazmats Post-NMS: {len(final_hazmat_detections)}")
    return final_hazmat_detections # Return list of (label, box)

# --- END OF FILE orb_hazmat_detector.py ---