def filter_detections(detections, model, object_classes):
    filtered_detections = []
    for detection in detections:
        class_id = int(detection[5])  # Assuming the class ID is at index 5
        class_name = model.names[class_id]
        if class_name in object_classes:
            filtered_detections.append(detection)
    return filtered_detections