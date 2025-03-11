from flask import Flask, render_template, request, redirect, url_for, Response, send_file, jsonify
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import numpy as np
import os
from scripts.detect_coco import detect_coco
from scripts.detect_material import detect_material
from utils.calculate_iou import calculate_iou
from models.coco_model import coco_model
from models.material_model import material_model
from playsound import playsound
import base64
import threading

app = Flask(__name__)

# Define the material classes for the custom model
material_classes = ['concrete', 'metal', 'plastic']

# Define the classes to exclude from material classification
excluded_classes = list(range(0, 24)) + list(range(46, 55)) + [64]

# Define high-risk COCO classes (including all animals)
high_risk_coco_classes = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

# Define classes to exclude from risk assignment
no_risk_classes = list(range(0, 9)) + list(range(46, 56))

ALERT_SOUND_PATH = r"D:\mini_coco\alert.mp3"
alarm_thread = None
alarm_thread_stop_event = threading.Event()

def classify_risk(class_name, material_class_name):
    if class_name in high_risk_coco_classes or material_class_name == 'metal':
        return 'High Risk'
    elif material_class_name == 'concrete':
        return 'Medium Risk'
    else:
        return 'Low Risk'

def play_alarm():
    while not alarm_thread_stop_event.is_set():
        playsound(ALERT_SOUND_PATH)

def stop_alarm():
    global alarm_thread
    if alarm_thread and alarm_thread.is_alive():
        alarm_thread_stop_event.set()
        alarm_thread.join()
        alarm_thread_stop_event.clear()

def detect_and_classify(image_path):
    # Run object detection using the COCO model
    coco_detections = detect_coco(image_path)

    # Run material classification using the custom model
    material_detections = detect_material(image_path)

    # Match results using IoU and annotate the image
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    high_risk_detected = False
    high_risk_details = []

    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        material_class_name = 'unknown'
        risk_level = 'No Risk' if class_id in no_risk_classes else classify_risk(class_name, material_class_name)

        # Check if the class_id is not in the excluded classes
        if class_id not in excluded_classes:
            # Check for overlapping material detections
            for material_detection in material_detections:
                mx1, my1, mx2, my2, mconf, mclass_id = material_detection[:6]
                detected_material_class_name = material_classes[int(mclass_id)]
                iou = calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
                if iou > 0.5:  # Consider as overlapping if IoU > 0.5
                    material_class_name = detected_material_class_name
                    risk_level = 'No Risk' if class_id in no_risk_classes else classify_risk(class_name, material_class_name)
                    ax.text(x1, y2, f'{material_class_name} {mconf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
                    break

        if conf >= 0.4:
            ax.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        if risk_level == 'High Risk':
            high_risk_detected = True
            high_risk_details.append({'class_name': class_name, 'confidence': conf})
            global alarm_thread
            if not alarm_thread or not alarm_thread.is_alive():
                alarm_thread = threading.Thread(target=play_alarm)
                alarm_thread.start()
        
        if risk_level != 'No Risk':
            ax.text(x1, y2 + 20, f'Risk: {risk_level}', color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)
    
    return img_bytes, high_risk_detected, high_risk_details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Ensure the uploads directory exists
        os.makedirs('uploads', exist_ok=True)
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        img_bytes, high_risk_detected, high_risk_details = detect_and_classify(image_path)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        response = {
            'image': img_base64,
            'high_risk_detected': high_risk_detected,
            'high_risk_details': high_risk_details
        }
        return jsonify(response)

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    stop_alarm()
    return jsonify({'status': 'alarm stopped'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Ensure the uploads directory exists
        os.makedirs('uploads', exist_ok=True)
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)
        return redirect(url_for('video_feed', video_path=video_path))

def process_frame(frame):
    # Convert the frame to an image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run object detection using the COCO model
    coco_detections = detect_coco(image)

    # Run material classification using the custom model
    material_detections = detect_material(image)

    # Match results using IoU and annotate the image
    image_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)

    high_risk_detected = False

    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        material_class_name = 'unknown'
        risk_level = 'No Risk' if class_id in no_risk_classes else classify_risk(class_name, material_class_name)

        # Check if the class_id is not in the excluded classes
        if class_id not in excluded_classes:
            # Check for overlapping material detections
            for material_detection in material_detections:
                mx1, my1, mx2, my2, mconf, mclass_id = material_detection[:6]
                detected_material_class_name = material_classes[int(mclass_id)]
                iou = calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
                if iou > 0.5:  # Consider as overlapping if IoU > 0.5
                    material_class_name = detected_material_class_name
                    risk_level = 'No Risk' if class_id in no_risk_classes else classify_risk(class_name, material_class_name)
                    ax.text(x1, y2, f'{material_class_name} {mconf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
                    break

        if conf >= 0.4:
            ax.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        if risk_level == 'High Risk':
            high_risk_detected = True
            global alarm_thread
            if not alarm_thread or not alarm_thread.is_alive():
                alarm_thread = threading.Thread(target=play_alarm)
                alarm_thread.start()
        
        if risk_level != 'No Risk':
            ax.text(x1, y2 + 20, f'Risk: {risk_level}', color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)
    return img_bytes

def gen_frames(video_path=None):
    if video_path:
        cap = cv2.VideoCapture(video_path)  # Use the video file path
    else:
            processed_frame = process_frame(frame)
            frame = cv2.imdecode(np.frombuffer(processed_frame, np.uint8), cv2.IMREAD_COLOR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)