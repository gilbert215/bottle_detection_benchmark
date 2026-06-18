#!/usr/bin/env python3
"""
Train and evaluate YOLOv8 on the bottle dataset using Ultralytics.

Run ``scripts/convert_coco_to_yolo.py`` first to produce the
``dataset.yaml`` this script expects.

Example
-------
    python scripts/train_yolov8.py --data data/bottles_yolo/dataset.yaml \
        --model yolov8n.pt --epochs 50 --imgsz 640
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base checkpoint (n/s/m/l/x)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/yolov8_bottles")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
    )

    # Validate on the val/test split defined in dataset.yaml and print the
    # standard Ultralytics metrics (mAP50, mAP50-95, precision, recall, speed).
    metrics = model.val(data=args.data)
    print(metrics)


if __name__ == "__main__":
    main()
