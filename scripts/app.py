from flask import Flask, render_template, request, redirect, url_for, Response, send_file, jsonify
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import numpy as np
import os
from scripts.detect_coco import detect_coco
from models.coco_model import coco_model
import base64

app = Flask(__name__)

# Define the classes to include in detection
included_classes = list(range(0, 10)) + list(range(14, 24)) + [63, 66, 67, 73, 74, 76]

def detect_coco(image_path):
    results = coco_model(image_path)
    detections = results[0].boxes.data.tolist()
    detections = [d for d in detections if int(d[5]) in included_classes]
    return detections

def detect_and_annotate(image_path):
    coco_detections = detect_coco(image_path)
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)
    
    return img_bytes

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
        os.makedirs('uploads', exist_ok=True)
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        img_bytes = detect_and_annotate(image_path)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({'image': img_base64})

def process_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    coco_detections = detect_coco(image)
    image_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)

    for coco_detection in coco_detections:
        x1, y1, x2, y2, conf, class_id = coco_detection[:6]
        class_name = coco_model.names[int(class_id)]
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{class_name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)

    processed_frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return processed_frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
