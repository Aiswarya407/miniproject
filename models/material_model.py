from ultralytics import YOLO

# Load the custom material YOLOv8 model
material_model = YOLO(r'D:\mini_coco\runs\detect\train4\weights\best.pt')