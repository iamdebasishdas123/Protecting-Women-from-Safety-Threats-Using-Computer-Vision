from ultralytics import YOLO

# Load YOLOv8 model for person detection
yolo_model = YOLO("yolov8n.pt")

def detect_person(frame):
    results = yolo_model(frame)
    person_boxes = []
    print(" detect_person")
    if len(results) > 0:
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    if box.cls == 0:  # 'person' class index
                        x1, y1, x2, y2 = box.xyxy[0]
                        person_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return person_boxes
