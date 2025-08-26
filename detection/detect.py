import argparse
import cv2
import os
from ultralytics import YOLO

def filter_classes(target_classes, model_classes):
    if target_classes:
        invalid_classes = [c for c in target_classes if c.lower().replace("-", " ") not in model_classes]
        if invalid_classes:
            raise ValueError(f"[Error] The following classes are not in the model: {invalid_classes}\n"
                             f"[Info] Available classes: {model_classes}")
        print(f"[Info] Filtering detections by classes: {target_classes}")

        target_classes = [c.lower().replace(" ", "-") for c in target_classes]

def main(image_path, model_path="yolov8x-obb.onnx", output_path="result.png", target_classes=None):
    model = YOLO(model_path)
    class_names = model.names

    filter_classes(target_classes, list(class_names.values()))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[Error] Image not found: {image_path}")

    results = model.predict(img, imgsz=640, conf=0.3)

    for result in results:
        if target_classes:
            filtered_obb = [box for box in result.obb
                            if box.cls is not None and
                            result.names[int(box.cls)].lower().replace(" ", "-") in target_classes]
            result.obb = filtered_obb

    annotated_frame = result.plot()

    cv2.imwrite(output_path, annotated_frame)
    print(f"[Info] Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 OBB ONNX model on an image with class filtering")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="yolov8x-obb.onnx", help="Path to the YOLOv8 ONNX model")
    parser.add_argument("--output", default="result.png", help="Path to save the output image")
    parser.add_argument("--classes", nargs="+", help="List of class names to detect (e.g. --classes car plane ship)")
    args = parser.parse_args()

    main(args.image, args.model, args.output, args.classes)