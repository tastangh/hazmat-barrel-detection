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