import cv2

def draw_detections(frame, detections):
    for label, (x, y, w, h) in detections:
        color = (0, 255, 0)
        if "Varil" in label:
            color = (255, 0, 0) if "Mavi" in label else (0, 0, 255) 
        elif "Hazmat" in label or any(char.isdigit() for char in label): 
             color = (0, 255, 255) 

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(y, label_size[1] + 10)
        cv2.rectangle(frame, (x, y_label - label_size[1] - 10),
                      (x + label_size[0], y_label + base_line - 10), color, cv2.FILLED)

        cv2.putText(frame, label, (x, y_label - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame