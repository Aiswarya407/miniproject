import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import os
from models import coco_model, material_model
from utils import calculate_iou, filter_detections

# Define the material classes for the custom model
material_classes = ['plastic', 'metal', 'concrete']

# Perform object detection and material classification on an image
def detect_and_classify(image_path):
    # Run object detection using the COCO model
    coco_results = coco_model(image_path)
    coco_detections = coco_results[0].boxes.data.tolist()  # Access the detections directly

    # Run material classification using the custom model
    material_results = material_model(image_path)
    material_detections = material_results[0].boxes.data.tolist()  # Access the detections directly

    # Match results using IoU and annotate the image
    image = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        # Check for overlapping material detections
        for material_detection in material_detections:
            mx1, my1, mx2, my2, mconf, mclass_id = material_detection[:6]
            material_class_name = material_classes[int(mclass_id) - 80]  # Adjusting class_id to match material_classes
            iou = calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
            if iou > 0.5:  # Consider as overlapping if IoU > 0.5
                plt.text(x1, y2, f'{material_class_name} {mconf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

    plt.axis('off')
    plt.show()

    # Save the annotated image
    output_dir = 'outputs/combined_results/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'annotated_image.jpg')
    plt.savefig(output_path)
    print(f"Annotated image saved to {output_path}")

# Example usage
image_path = r'D:\mini_coco\testimage1.webp'  # Replace with the path to your image
detect_and_classify(image_path)