"""
Batch-process all VisDrone sequences.
Usage: python scripts/run_all_sequences.py --dataset <path> --output outputs/
"""
import argparse, os, subprocess, sys

def run_all(args):
    os.makedirs(args.output, exist_ok=True)
    sequences = sorted([d for d in os.listdir(args.dataset)
                        if os.path.isdir(os.path.join(args.dataset, d))])
    print(f"Found {len(sequences)} sequences.\n")

    for i, seq in enumerate(sequences, 1):
        print(f"[{i}/{len(sequences)}] {seq}")
        subprocess.run([
            sys.executable, "src/run_pipeline.py",
            "--source", os.path.join(args.dataset, seq),
            "--output", os.path.join(args.output, f"{seq}_result.mp4"),
            "--config", args.config,
            "--device", args.device,
        ])

    print(f"\nDone. Results in: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="outputs/")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--device", default="cpu")
    return p.parse_args()

if __name__ == "__main__":
    run_all(parse_args())
