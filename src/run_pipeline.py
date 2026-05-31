"""
run_pipeline.py — Main entry point for Aerial Guardian pipeline.

Usage:
    python src/run_pipeline.py --source <path> --output <output.mp4>
"""

import argparse
import time
import cv2
import yaml
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import DroneDetector
from src.tracker import ByteTracker
from src.ego_motion import EgoMotionCompensator
from src.visualizer import TrackVisualizer
from src.utils import get_video_info, ensure_dir


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(args):
    cfg = load_config(args.config)
    if args.device:
        cfg["detector"]["device"] = args.device
    if args.output:
        cfg["output"]["path"] = args.output

    print(f"\n{'='*60}")
    print(f"  Aerial Guardian — Drone MOT Pipeline")
    print(f"{'='*60}")
    print(f"  Source : {args.source}")
    print(f"  Device : {cfg['detector']['device']}")
    print(f"  SAHI   : {'ON' if cfg['sahi']['enabled'] else 'OFF'}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {args.source}")

    width, height, fps_in, total_frames = get_video_info(cap)
    print(f"  Video  : {width}x{height} @ {fps_in:.1f} FPS ({total_frames} frames)\n")

    out_path = cfg["output"]["path"]
    ensure_dir(os.path.dirname(out_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    detector = DroneDetector(cfg["detector"], cfg["sahi"])
    tracker = ByteTracker(cfg["tracker"])
    ego_comp = EgoMotionCompensator()
    visualizer = TrackVisualizer(cfg["visualization"])

    frame_idx = 0
    prev_gray = None
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_shift = ego_comp.estimate(prev_gray, curr_gray)
        prev_gray = curr_gray

        detections = detector.detect(frame)
        detections = ego_comp.compensate_detections(detections, cam_shift)
        tracks = tracker.update(detections, frame)

        elapsed = time.perf_counter() - t_start
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(current_fps)
        avg_fps = sum(fps_history[-30:]) / len(fps_history[-30:])

        vis_frame = visualizer.draw(frame, tracks, avg_fps, frame_idx)
        writer.write(vis_frame)

        if args.show:
            cv2.imshow("Aerial Guardian", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx:4d}/{total_frames} | FPS: {avg_fps:.1f} | Tracks: {len(tracks)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    overall_fps = sum(fps_history) / len(fps_history) if fps_history else 0
    print(f"\n{'='*60}")
    print(f"  Done! Output saved to: {out_path}")
    print(f"  Average FPS: {overall_fps:.1f}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Aerial Guardian — Drone MOT Pipeline")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
