# --- START OF FILE main.py ---

import cv2
import sys
import os
import argparse
import numpy as np # Needed for numpy usage if any direct calls were made (less likely now)
from detector.orb_hazmat_detector import detect_hazmats
from detector.barrel_detector import detect_barrels
from utils.draw_utils import draw_detections
from utils.iou_nms import calculate_iou # Import IoU calculation

DETECTED_OBJECTS = set()

# --- Configuration ---
HAZMAT_DETECTION_INTERVAL = 3 # Detect hazmats every N frames (reduces computation)
IOU_THRESHOLD_BARREL_HAZMAT = 0.2 # If IoU between barrel and hazmat > this, discard barrel
PAUSE_ON_NEW_DETECTION = True # Pause video when a new object type is detected
# --- End Configuration ---


def setup_logger(log_path="log.txt"):
    """Sets up logging to both console and file."""
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            # Use 'a' for append mode, 'w' for overwrite
            self.log = open(log_path, "a", encoding="utf-8", buffering=1) # Line buffering

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            # self.flush() # Flushing frequently can impact performance

        def flush(self):
            # This flush method is needed for compatibility.
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    # Keep track of the original stdout
    original_stdout = sys.stdout
    logger = Logger()
    sys.stdout = logger
    return logger, original_stdout


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video Properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    frame_idx = 0
    hazmats = [] # Store last detected hazmats

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] End of video stream.")
            break

        print(f"\n[Frame {frame_idx}] -------------------------------")

        # Detect barrels in every frame
        barrels = detect_barrels(frame)
        # print(f"[DEBUG] Raw Barrel Detections: {len(barrels)}")

        # Detect hazmats periodically
        if frame_idx % HAZMAT_DETECTION_INTERVAL == 0:
            hazmats = detect_hazmats(frame) # Returns NMS-filtered list: [(label, box)]
            # print(f"[DEBUG] Hazmat Detections (every {HAZMAT_DETECTION_INTERVAL} frames): {len(hazmats)}")
        # else: use hazmats from previous detection interval for filtering barrels


        # --- Filter Barrels Overlapping with Hazmats ---
        final_barrels = []
        hazmat_boxes = [h[1] for h in hazmats] # Get just the boxes (x,y,w,h)

        for barrel_label, barrel_box in barrels:
            is_overlapping_with_hazmat = False
            for hazmat_box in hazmat_boxes:
                iou = calculate_iou(barrel_box, hazmat_box)
                if iou > IOU_THRESHOLD_BARREL_HAZMAT:
                    is_overlapping_with_hazmat = True
                    # print(f"[!] Discarding '{barrel_label}' detection (IoU: {iou:.2f}) overlapping with hazmat @ {hazmat_box}")
                    break # No need to check other hazmats for this barrel
            if not is_overlapping_with_hazmat:
                final_barrels.append((barrel_label, barrel_box))
            # else: already printed discard message if needed

        # print(f"[DEBUG] Final Barrel Detections after filtering: {len(final_barrels)}")

        # Combine filtered barrels and current hazmats
        # Hazmats list already contains (label, box) tuples
        combined_detections = hazmats + final_barrels

        # --- Track and Announce New Detections ---
        new_detections_this_frame = []
        for label, box in combined_detections:
            # Normalize label for tracking (e.g., handle minor variations if needed)
            normalized_label = label # Simple case for now
            if normalized_label not in DETECTED_OBJECTS:
                DETECTED_OBJECTS.add(normalized_label)
                print(f"[*] >>> New Object Type Detected: {normalized_label} @ Frame {frame_idx}")
                new_detections_this_frame.append((label, box))


        # --- Draw Detections and Display ---
        # Frame numarasını videoya yaz
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw all combined detections (filtered barrels + hazmats)
        frame_display = draw_detections(frame.copy(), combined_detections) # Draw on a copy
        cv2.imshow("Detections", frame_display)

        # --- User Control ---
        wait_time = 0 if (PAUSE_ON_NEW_DETECTION and new_detections_this_frame) else 1
        key = cv2.waitKey(wait_time) & 0xFF # Use '& 0xFF' for 64-bit compatibility

        if key == ord('q'):
            print("\n[INFO] 'q' pressed. Exiting.")
            break
        elif key == ord('p'): # Pause toggle
             print("[INFO] Paused. Press any key (except 'q') to resume.")
             while True:
                 key_resume = cv2.waitKey(0) & 0xFF
                 if key_resume == ord('q'):
                     key = ord('q') # Propagate quit signal
                     break
                 if key_resume is not None:
                     print("[INFO] Resumed.")
                     break
             if key == ord('q'): # Check if quit was pressed during pause
                 break
        elif key == ord(' '): # Advance frame by frame when paused
            pass # Handled by waitKey(0) when paused

        frame_idx += 1
        # --- End Main Loop ---

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Hazmat placards and Barrels in a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., ./data/tusas-odev1.mp4)")
    parser.add_argument("--log", type=str, default="logs/output.log", help="Path to the output log file.")
    args = parser.parse_args()

    # Validate video path
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    # Setup logger
    logger, original_stdout = setup_logger(args.log)
    print("--- Detection Script Started ---")
    print(f"Input Video: {args.video}")
    print(f"Log File: {args.log}")
    print(f"Hazmat Detection Interval: {HAZMAT_DETECTION_INTERVAL} frames")
    print(f"Barrel/Hazmat Overlap Threshold (IoU): {IOU_THRESHOLD_BARREL_HAZMAT}")
    print(f"Pause on New Detection: {PAUSE_ON_NEW_DETECTION}")
    print("--------------------------------")


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