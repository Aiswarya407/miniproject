from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Print all COCO classes
print(model.names)
