from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'D:\mini_coco\runs\detect\train\weights\last.pt')  # You can start with a pre-trained model

# Train the model with the custom dataset
model.train(data='custom_dataset.yaml', epochs=10, imgsz=640)

# Save the trained model
model.save('best.pt')
