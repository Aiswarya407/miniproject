import os
from scripts.train_material import train_material_model
from scripts.combine_models import detect_and_classify

def main():
    # Step 1: Train the material classification model
    print("Training the material classification model...")
    train_material_model()

    # Step 2: Perform object detection and material classification
    image_path = r'D:\mini_coco\testimage1.webp'  # Replace with the path to your image
    print(f"Running detection and classification on {image_path}...")
    detect_and_classify(image_path)

if __name__ == "__main__":
    main()