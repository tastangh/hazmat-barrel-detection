# --- START OF FILE main.py ---

import cv2
import sys
import os
import argparse
import numpy as np
import time
from collections import deque

# Use the ORB detector now
from detector.orb_hazmat_detector import detect_hazmats
from detector.barrel_detector import detect_barrels
from utils.draw_utils import draw_detections
from utils.iou_nms import calculate_iou

# --- Configuration ---
# Performance
RESIZE_FACTOR = 0.75 # 1.0 = no resize. 0.75 = 75% size. 0.5 = 50% size. Smaller = faster but less detail.
HAZMAT_DETECTION_INTERVAL = 5 # Detect hazmats every N frames. Higher = faster. Lower = better tracking. Adjust based on speed/needs.

# Filtering & Tracking
IOU_THRESHOLD_BARREL_HAZMAT = 0.2 # If IoU between barrel and hazmat > this, discard barrel
PAUSE_ON_NEW_DETECTION = True     # Pause video when a new object type is *confirmed*

# Temporal Consistency Parameters
TEMPORAL_IOU_THRESHOLD = 0.4       # Min IoU to associate detection with existing track
TEMPORAL_MAX_AGE = 8              # Max frames a track can live without being updated (increase if HAZMAT_DETECTION_INTERVAL is high)
TEMPORAL_MIN_HITS_TO_CONFIRM = 3  # Min consecutive/nearby hits needed to confirm a new track/label change
TEMPORAL_CONFIDENCE_THRESHOLD = 10 # Min confidence from detector (RANSAC inliers) to consider a detection
# --- End Configuration ---


# --- Globals ---
DETECTED_OBJECTS = set() # Keep track of unique *confirmed* labels ever seen
active_hazmat_tracks = []
next_track_id = 0
# --- End Globals ---


# --- Logger Setup ---
def setup_logger(log_path="log.txt"):
    """Sets up logging to both console and file."""
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path, "a", encoding="utf-8", buffering=1) # Append mode, line buffering

        def write(self, message):
            try:
                self.terminal.write(message)
                self.log.write(message)
            except Exception as e: # Catch potential encoding errors on exotic terminals
                 self.terminal.write(f"[Logger Error: {e}]\n")


        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    original_stdout = sys.stdout
    logger = Logger()
    sys.stdout = logger
    return logger, original_stdout
# --- End Logger Setup ---


# --- Tracking Class ---
class HazmatTrack:
    def __init__(self, track_id, label, box, confidence, frame_idx):
        self.id = track_id
        self.label = label      # Current best label
        self.box = box          # Current best box (x,y,w,h)
        self.confidence = confidence # Confidence of the last match
        self.age = 0            # Frames since last update
        self.hits = 1           # Number of matches (used for confirmation)
        self.last_seen_frame = frame_idx
        # Store history for label stability checks
        self.label_history = deque([label], maxlen=max(5, TEMPORAL_MIN_HITS_TO_CONFIRM + 2))
        self.is_confirmed = False # Track confirmation status

    def update(self, label, box, confidence, frame_idx):
        self.box = box
        self.confidence = confidence
        self.age = 0 # Reset age on update
        self.last_seen_frame = frame_idx
        self.hits += 1

        # --- Label Stability Logic ---
        self.label_history.append(label)
        # Check if the most frequent label in recent history is different from current
        if len(self.label_history) > 1:
            most_common_label = max(set(self.label_history), key=list(self.label_history).count)
            if most_common_label != self.label:
                # Change label only if it's consistently different
                new_label_count = sum(1 for lbl in self.label_history if lbl == most_common_label)
                # Require a majority or strong trend to change confirmed label
                if new_label_count >= max(TEMPORAL_MIN_HITS_TO_CONFIRM -1, len(self.label_history) // 2 + 1):
                     if self.label != most_common_label:
                         print(f"[Track {self.id}] Label changed {self.label} -> {most_common_label} (History: {list(self.label_history)}, Hits: {self.hits})")
                         self.label = most_common_label

        # Update confirmation status
        if not self.is_confirmed and self.hits >= TEMPORAL_MIN_HITS_TO_CONFIRM:
            self.is_confirmed = True
            print(f"[Track {self.id}] Confirmed as {self.label} (Hits: {self.hits})")


    def predict_box(self):
        # Simple prediction: return last known box.
        # TODO: Implement Kalman Filter for smoother prediction if needed.
        return self.box
# --- End Tracking Class ---


# --- Main Function ---
def main(video_path):
    global active_hazmat_tracks, next_track_id, DETECTED_OBJECTS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    print(f"[INFO] Processing at scale: {RESIZE_FACTOR*100:.0f}%")
    print(f"[INFO] Hazmat Detection Interval: {HAZMAT_DETECTION_INTERVAL} frames")

    frame_idx = 0
    processing_times = deque(maxlen=int(fps if fps > 0 else 30)) # Track time for ~1 sec

    while cap.isOpened():
        start_time = time.time()

        ret, frame_orig = cap.read()
        if not ret:
            print("\n[INFO] End of video stream.")
            break

        # 1. Resize Frame (if needed)
        if RESIZE_FACTOR != 1.0:
            frame = cv2.resize(frame_orig, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_LINEAR)
        else:
            frame = frame_orig.copy() # Process on a copy if not resizing

        # 2. Increment Age for all tracks before detection
        for track in active_hazmat_tracks:
             track.age += 1

        # 3. Get Raw Detections
        # Barrels detected on every frame (usually fast enough)
        # Coordinates are relative to the potentially resized 'frame'
        barrels = detect_barrels(frame)

        # Hazmats detected periodically
        hazmats_confident = []
        if frame_idx % HAZMAT_DETECTION_INTERVAL == 0:
            # detect_hazmats returns: [(label, box, score)]
            hazmats_raw = detect_hazmats(frame)
            # Filter detections below confidence threshold *before* tracking
            hazmats_confident = [det for det in hazmats_raw if det[2] >= TEMPORAL_CONFIDENCE_THRESHOLD]
            # print(f"[DEBUG] Frame {frame_idx}: Raw={len(hazmats_raw)}, Confident={len(hazmats_confident)}")


        # 4. Match Detections to Tracks (using confident detections)
        matched_track_indices = set()
        unmatched_detection_indices = list(range(len(hazmats_confident)))
        track_indices_to_match = list(range(len(active_hazmat_tracks)))

        if hazmats_confident and active_hazmat_tracks: # Only match if both exist
            iou_matrix = np.zeros((len(active_hazmat_tracks), len(hazmats_confident)), dtype=np.float32)

            for t, track_idx in enumerate(track_indices_to_match):
                track = active_hazmat_tracks[track_idx]
                track_box = track.predict_box()
                for d, det_idx in enumerate(unmatched_detection_indices):
                    det_box = hazmats_confident[det_idx][1]
                    iou_matrix[t, d] = calculate_iou(track_box, det_box)

            # Greedy matching based on IoU (or use Hungarian algorithm for optimal)
            matched_det_indices = set()
            matches_potential = []
            for t in range(len(active_hazmat_tracks)):
                for d in range(len(hazmats_confident)):
                    if iou_matrix[t, d] >= TEMPORAL_IOU_THRESHOLD:
                        matches_potential.append((iou_matrix[t, d], t, d))

            matches_potential.sort(key=lambda x: x[0], reverse=True) # Sort by IoU descending

            for iou, track_rel_idx, det_idx in matches_potential:
                track_abs_idx = track_indices_to_match[track_rel_idx] # Get original index
                if track_abs_idx not in matched_track_indices and det_idx not in matched_det_indices:
                    track = active_hazmat_tracks[track_abs_idx]
                    label, box, score = hazmats_confident[det_idx]
                    track.update(label, box, score, frame_idx)
                    matched_track_indices.add(track_abs_idx)
                    matched_det_indices.add(det_idx)


        # 5. Handle Unmatched Tracks (Remove old ones)
        tracks_to_keep = []
        for i, track in enumerate(active_hazmat_tracks):
            if i in matched_track_indices:
                tracks_to_keep.append(track)
            else: # Unmatched this frame
                if track.age <= TEMPORAL_MAX_AGE:
                    tracks_to_keep.append(track) # Keep it if not too old
                else:
                    print(f"[Track {track.id}] Removed ({track.label}, Age: {track.age})")
        active_hazmat_tracks = tracks_to_keep


        # 6. Create New Tracks for Unmatched Detections
        for det_idx in range(len(hazmats_confident)):
            if det_idx not in matched_det_indices:
                label, box, score = hazmats_confident[det_idx]
                new_track = HazmatTrack(next_track_id, label, box, score, frame_idx)
                active_hazmat_tracks.append(new_track)
                print(f"[Track {next_track_id}] Created ({label}, Score: {score:.1f})")
                next_track_id += 1


        # 7. Generate Final Detections List from Confirmed Tracks
        final_hazmat_detections_for_display = []
        newly_confirmed_labels_this_frame = set()

        for track in active_hazmat_tracks:
            if track.is_confirmed:
                # Add track ID to label for display uniqueness
                display_label = f"{track.label}-{track.id}"
                final_hazmat_detections_for_display.append((display_label, track.box))

                # Check if this confirmation is the first time seeing this *label*
                if track.label not in DETECTED_OBJECTS:
                    DETECTED_OBJECTS.add(track.label)
                    newly_confirmed_labels_this_frame.add(track.label)
                    print(f"[*] >>> New Object Type Confirmed: {track.label} @ Frame {frame_idx} (Track {track.id})")


        # 8. Filter Barrels Overlapping with Confirmed Hazmats
        final_barrels = []
        confirmed_hazmat_boxes = [h[1] for h in final_hazmat_detections_for_display] # Use boxes from confirmed tracks

        for barrel_label, barrel_box in barrels:
            is_overlapping = False
            for hazmat_box in confirmed_hazmat_boxes:
                if calculate_iou(barrel_box, hazmat_box) > IOU_THRESHOLD_BARREL_HAZMAT:
                    is_overlapping = True
                    break
            if not is_overlapping:
                final_barrels.append((barrel_label, barrel_box))


        # 9. Combine Final Detections for Display
        # Coordinates are still relative to the resized 'frame'
        combined_detections_final = final_hazmat_detections_for_display + final_barrels


        # 10. Draw Detections and Display
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        avg_fps = len(processing_times) / sum(processing_times) if processing_times else 0

        # Draw results on the *original* frame, scaling coordinates back
        frame_display = draw_detections(frame_orig, combined_detections_final, scale_factor=RESIZE_FACTOR, draw_track_id=True)

        # Add Frame# and FPS to display
        cv2.putText(frame_display, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, f"FPS: {avg_fps:.1f}", (frame_display.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Detections", frame_display)


        # 11. User Control
        pause_now = PAUSE_ON_NEW_DETECTION and bool(newly_confirmed_labels_this_frame)
        wait_time = 0 if pause_now else 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            print("\n[INFO] 'q' pressed. Exiting.")
            break
        elif key == ord('p') or (pause_now and key != ord(' ')): # Pause toggle or automatic pause
             print("[INFO] Paused. Press SPACE to advance frame, 'p' to resume freely, 'q' to quit.")
             while True:
                 key_resume = cv2.waitKey(0) & 0xFF
                 if key_resume == ord('q'):
                     key = ord('q'); break
                 elif key_resume == ord(' '): # Advance one frame
                     break
                 elif key_resume == ord('p'): # Resume fully
                     print("[INFO] Resumed.")
                     break
             if key == ord('q'): break # Exit if q was pressed during pause

        frame_idx += 1
        # --- End Main Loop ---

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released.")
# --- End Main Function ---


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Hazmat placards and Barrels in a video with ORB and Temporal Consistency.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., ./data/your_video.mp4)")
    parser.add_argument("--log", type=str, default="logs/output.log", help="Path to the output log file.")
    parser.add_argument("--resize", type=float, default=RESIZE_FACTOR, help=f"Frame resize factor (e.g., 0.75). Default={RESIZE_FACTOR}")
    parser.add_argument("--interval", type=int, default=HAZMAT_DETECTION_INTERVAL, help=f"Hazmat detection interval (frames). Default={HAZMAT_DETECTION_INTERVAL}")

    args = parser.parse_args()

    # Override defaults from command line if provided
    RESIZE_FACTOR = args.resize
    HAZMAT_DETECTION_INTERVAL = args.interval

    # Validate video path
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    # Setup logger
    logger, original_stdout = setup_logger(args.log)
    print("--- Detection Script Started (ORB Detector, Temporal Tracking) ---")
    print(f"Input Video: {args.video}")
    print(f"Log File: {args.log}")
    print(f"Resize Factor: {RESIZE_FACTOR}")
    print(f"Hazmat Detection Interval: {HAZMAT_DETECTION_INTERVAL}")
    print(f"Barrel/Hazmat Overlap Threshold (IoU): {IOU_THRESHOLD_BARREL_HAZMAT}")
    print(f"Pause on New Confirmed Detection: {PAUSE_ON_NEW_DETECTION}")
    print(f"Temporal IoU Threshold: {TEMPORAL_IOU_THRESHOLD}")
    print(f"Temporal Max Age: {TEMPORAL_MAX_AGE}")
    print(f"Temporal Min Hits to Confirm: {TEMPORAL_MIN_HITS_TO_CONFIRM}")
    print(f"Temporal Min Confidence: {TEMPORAL_CONFIDENCE_THRESHOLD}")
    print("--------------------------------------------------------------")
    print("--- ORB Detector Parameters (Needs Tuning!) ---")
    print(f"ORB Max Features: {ORB_MAX_FEATURES}")
    print(f"Min Good Matches: {MIN_GOOD_MATCHES}")
    print(f"Min RANSAC Inliers: {MIN_RANSAC_INLIERS}")
    print(f"Min Box Size: {MIN_BOX_SIZE}")
    print(f"Min Confidence (Inliers): {MIN_CONFIDENCE}")
    print(f"Aspect Ratio Range: {MIN_ASPECT_RATIO:.1f}-{MAX_ASPECT_RATIO:.1f}")
    print(f"NMS Threshold: {NMS_THRESHOLD_HAZMAT}")
    print("--------------------------------------------------------------")


    try:
        main(args.video)
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console/log
    finally:
        # Restore original stdout and close logger
        sys.stdout = original_stdout
        if hasattr(logger, 'close'):
            logger.close()
        print("--- Detection Script Finished ---")


# --- END OF FILE main.py ---