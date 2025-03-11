from ultralytics import YOLO

# Load the model from the previous checkpoint
model = YOLO(r'D:\mini_coco\runs\detect\train2\weights\last.pt')

# Continue training, increasing the total epochs
model.train(data='custom_dataset.yaml', epochs=50, imgsz=640)
# Save the trained model

model.save('best.pt')
