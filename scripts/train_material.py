from ultralytics import YOLO

def train_material_model():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can start with a pre-trained model

    # Train the model with the custom dataset
    model.train(data='custom_dataset.yaml', epochs=20, imgsz=640)

    # Save the trained model
    model.save('best.pt')

if __name__ == "__main__":
    train_material_model()