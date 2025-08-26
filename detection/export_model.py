import argparse
import os
from ultralytics import YOLO    # pip install ultralytics


def main(dest_dir_name="models", model_name="yolov8x-obb.pt"):
    if not os.path.exists(dest_dir_name) and dest_dir_name != "":
        os.makedirs(dest_dir_name)

    # Pretrained YOLOv8 OBB model (trained on DOTA dataset)
    print(f"[Info] Loading model: {model_name}")
    model = YOLO(model_name)

    export_path = os.path.join(dest_dir_name, os.path.splitext(os.path.basename(model_name))[0] + ".onnx")
    print(f"[Info] Exporting to ONNX: {export_path}")
    model.export(format="onnx", imgsz=640, dynamic=True) # specify additional arguments if needed

    print("\n[Info] Model architecture summary:")
    print(model.model)

    if hasattr(model.model, "stride"):
        print(f"\n[Info] Model stride: {model.model.stride}")
    if hasattr(model.model, "names"):
        print(f"[Info] Model classes: {model.model.names}")

    print(f"\n[Info] ONNX model exported to: {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 OBB model to ONNX")
    parser.add_argument("--dest_dir", default="models", help="Destination directory for the YOLOv8 ONNX model")
    parser.add_argument("--model", default="yolov8x-obb.pt", help="Source YOLOv8 PyTorch model")
    args = parser.parse_args()

    main(args.dest_dir, args.model)
