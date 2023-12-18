from ultralytics import YOLO
import os


if __name__ == '__main__': 
    model = YOLO('yolov8s.yaml')
    epochs = 200
    imgsz = 380
    results = model.train(
        data = "yolo_data/data.yml",
        epochs = epochs,
        imgsz = imgsz,
        project = 'models',
        name = "yolov8/detect/train"
    )