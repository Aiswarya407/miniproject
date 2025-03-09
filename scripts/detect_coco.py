from models.coco_model import coco_model

def detect_coco(image_path):
    results = coco_model(image_path)
    detections = results[0].boxes.data.tolist()  # Access the detections directly
    return detections