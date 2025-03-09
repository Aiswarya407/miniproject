import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from scripts.detect_coco import detect_coco
from scripts.detect_material import detect_material
from utils.calculate_iou import calculate_iou
from models.coco_model import coco_model

# Define the material classes for the custom model
material_classes = ['concrete', 'metal', 'plastic']

# Define the classes to exclude from material classification
excluded_classes = list(range(0, 24)) + list(range(46, 56)) + [64]

# Define high-risk COCO classes (including all animals)
high_risk_coco_classes = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

def classify_risk(class_name, material_class_name, area):
    if class_name in high_risk_coco_classes or material_class_name == 'metal' or area > 50000:
        return 'High Risk'
    elif material_class_name == 'concrete' or (10000 <= area <= 50000):
        return 'Medium Risk'
    else:
        return 'Low Risk'

def detect_and_classify(image_path):
    # Run object detection using the COCO model
    coco_detections = detect_coco(image_path)

    # Run material classification using the custom model
    material_detections = detect_material(image_path)

    # Match results using IoU and annotate the image
    image = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        area = width * height
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        material_class_name = 'unknown'
        risk_level = classify_risk(class_name, material_class_name, area)

        # Check if the class_id is not in the excluded classes
        if class_id not in excluded_classes:
            # Check for overlapping material detections
            for material_detection in material_detections:
                mx1, my1, mx2, my2, mconf, mclass_id = material_detection[:6]
                detected_material_class_name = material_classes[int(mclass_id)]
                iou = calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
                if iou > 0.5:  # Consider as overlapping if IoU > 0.5
                    material_class_name = detected_material_class_name
                    risk_level = classify_risk(class_name, material_class_name, area)
                    plt.text(x1, y2, f'{material_class_name} {mconf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
                    break

        if conf >= 0.8:
            plt.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        plt.text(x1, y2 + 20, f'Risk: {risk_level}', color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    plt.axis('off')
    plt.show()

    # Save the annotated image
    output_dir = 'outputs/combined_results/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'annotated_image.jpg')
    plt.savefig(output_path)
    print(f"Annotated image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = r'D:\mini_coco\testimages\metal_test.jpg'  # Replace with the path to your image
    detect_and_classify(image_path)