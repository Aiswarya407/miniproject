from flask import Flask, render_template, Response
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import numpy as np
from scripts.detect_coco import detect_coco
from scripts.detect_material import detect_material
from utils.calculate_iou import calculate_iou
from models.coco_model import coco_model
from models.material_model import material_model

app = Flask(__name__)

# Define the material classes for the custom model
material_classes = ['concrete', 'metal', 'plastic']

# Define the classes to exclude from material classification
excluded_classes = list(range(0, 24)) + list(range(46, 56)) + [64]

# Define high-risk COCO classes (including all animals)
high_risk_coco_classes = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

def classify_risk(class_name, material_class_name):
    if class_name in high_risk_coco_classes or material_class_name == 'metal':
        return 'High Risk'
    elif material_class_name == 'concrete':
        return 'Medium Risk'
    else:
        return 'Low Risk'

def get_color_for_risk(risk_level):
    if risk_level == 'High Risk':
        return 'red'
    elif risk_level == 'Medium Risk':
        return 'orange'
    else:
        return 'yellow'

def process_frame(frame):
    # Convert the frame to an image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run object detection using the COCO model
    coco_detections = detect_coco(image)

    # Run material classification using the custom model
    material_detections = detect_material(image)

    # Match results using IoU and annotate the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Process COCO detections
    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        material_class_name = 'unknown'
        risk_level = classify_risk(class_name, material_class_name)
        color = get_color_for_risk(risk_level)
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Check if the class_id is not in the excluded classes
        if class_id not in excluded_classes:
            # Check for overlapping material detections
            for material_detection in material_detections:
                mx1, my1, mx2, my2, mconf, mclass_id = material_detection[:6]
                detected_material_class_name = material_classes[int(mclass_id)]
                iou = calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
                if iou > 0.5:  # Consider as overlapping if IoU > 0.3
                    material_class_name = detected_material_class_name
                    risk_level = classify_risk(class_name, material_class_name)
                    color = get_color_for_risk(risk_level)
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    plt.text(x1, y2, f'{material_class_name} {mconf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
                    break

        if conf >= 0.8:
            plt.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        plt.text(x1, y2 + 20, f'Risk: {risk_level}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    # Process material detections independently
    for material_detection in material_detections:
        mx1, my1, mx2, my2, mconf, mclass_id = material_detection[:6]
        detected_material_class_name = material_classes[int(mclass_id)]
        if mclass_id not in excluded_classes:
            risk_level = classify_risk('unknown', detected_material_class_name)
            color = get_color_for_risk(risk_level)
            rect = patches.Rectangle((mx1, my1), mx2 - mx1, my2 - my1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            plt.text(mx1, my1, f'{detected_material_class_name} {mconf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
            plt.text(mx1, my2 + 20, f'Risk: {risk_level}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close()
    return img_bytes

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with the IP camera URL
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = process_frame(frame)
            frame = cv2.imdecode(np.frombuffer(processed_frame, np.uint8), cv2.IMREAD_COLOR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)