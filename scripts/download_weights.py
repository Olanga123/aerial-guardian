"""Download YOLOv8n weights. Usage: python scripts/download_weights.py"""
import os, sys

def download():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)
    os.makedirs("weights", exist_ok=True)
    print("Downloading YOLOv8n (~6 MB)...")
    model = YOLO("yolov8n.pt")
    model.save("weights/yolov8n.pt")
    print(f"Saved to weights/yolov8n.pt ({os.path.getsize('weights/yolov8n.pt')/1e6:.1f} MB)")

if __name__ == "__main__":
    download()
