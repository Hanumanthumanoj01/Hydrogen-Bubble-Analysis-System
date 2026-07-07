"""
train_yolo.py
=============
Trains a YOLOv8 bubble detector.

Run this on a machine with a GPU (or Google Colab's free tier — Runtime >
Change runtime type > GPU) once you have a real labeled dataset. Running it
on the placeholder 3-image dataset checked into this repo is only useful as
a smoke test to confirm the pipeline executes; the resulting weights are not
a usable model.

Usage:
    python train_yolo.py --data data.yaml --epochs 100 --imgsz 640 --batch 16
    python train_yolo.py --smoke-test          # tiny run, proves the pipeline works
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="base checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--smoke-test", action="store_true",
                    help="tiny/fast run (1 epoch, small image size) to verify the pipeline runs")
    args = p.parse_args()

    if args.smoke_test:
        args.epochs, args.imgsz, args.batch = 1, 320, 2
        print("SMOKE TEST MODE — proving the pipeline runs, not training a usable model.\n")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=20,
        project="runs",
        name="bubble_detector",
    )
    print("\nTraining complete. Best weights: runs/bubble_detector/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
