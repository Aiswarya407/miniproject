from models.material_model import material_model

def detect_material(image_path):
    results = material_model(image_path)
    detections = results[0].boxes.data.tolist()  # Access the detections directly
    return detections