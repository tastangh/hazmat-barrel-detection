# --- START OF FILE utils/iou_nms.py ---

import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in (x, y, w, h) format.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    # iou = intersection area / union area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou

def non_max_suppression(boxes_scores_labels, overlapThresh):
    """
    Applies Non-Maximum Suppression (NMS) to bounding boxes.
    Args:
        boxes_scores_labels: A list of tuples, where each tuple is
                             (label, (x, y, w, h), score).
        overlapThresh: The IoU threshold for suppression.
    Returns:
        A list of tuples (label, (x, y, w, h)) representing the kept boxes.
    """
    if len(boxes_scores_labels) == 0:
        return []

    # Sort by score in descending order
    boxes_scores_labels.sort(key=lambda x: x[2], reverse=True)

    # Separate boxes, scores, and labels
    labels = [item[0] for item in boxes_scores_labels]
    boxes = [item[1] for item in boxes_scores_labels] # (x, y, w, h)
    scores = [item[2] for item in boxes_scores_labels]

    # Convert boxes to (x1, y1, x2, y2) format for easier area calculation if needed
    # x2 = x1 + w, y2 = y1 + h
    np_boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
    pick_indices = [] # Indices of boxes to keep

    # Calculate areas
    x1 = np_boxes[:, 0]
    y1 = np_boxes[:, 1]
    x2 = np_boxes[:, 2]
    y2 = np_boxes[:, 3]
    area = (x2 - x1) * (y2 - y1) # Using w*h directly would also work

    # Get indices sorted by scores (already done by Python sort, but needed for looping)
    idxs = np.arange(len(scores))

    while len(idxs) > 0:
        # Grab the last index in the indexes list (highest score) and add it to picks
        last = len(idxs) - 1
        i = idxs[last]
        pick_indices.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        # Comparing the current highest score box (i) with all remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute the ratio of overlap
        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        overlap = intersection / union

        # Delete all indexes from the index list that have overlap greater than threshold
        idxs_to_delete = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        idxs = np.delete(idxs, idxs_to_delete)


    # Return only the boxes that were picked using the original indices
    final_detections = [(labels[i], boxes[i]) for i in pick_indices]
    return final_detections

# --- END OF FILE utils/iou_nms.py ---