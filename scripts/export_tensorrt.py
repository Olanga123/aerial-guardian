"""
Export YOLOv8n to TensorRT for NVIDIA Jetson.
Usage: python scripts/export_tensorrt.py --weights weights/yolov8n.pt
"""
import argparse, os

def export(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = args.weights.replace(".pt", ".engine")
    print(f"Exporting to TensorRT: {out}")
    model.export(format="engine", imgsz=args.imgsz, half=args.half, device=args.device, simplify=True)
    print(f"Done: {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"\nUpdate configs/config.yaml:\n  model: {out}\n  device: cuda")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    export(parse_args())
