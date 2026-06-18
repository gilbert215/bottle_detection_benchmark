#!/usr/bin/env python3
"""
Train SSD300-VGG16 or Faster R-CNN (ResNet50-FPN) on the bottle dataset.

Examples
--------
    python scripts/train.py --model ssd300_vgg16 --data-root data/bottles \
        --epochs 10 --batch-size 4 --lr 0.001 --output checkpoints/ssd.pth

    python scripts/train.py --model fasterrcnn_resnet50_fpn --data-root data/bottles \
        --epochs 10 --batch-size 4 --lr 0.001 --output checkpoints/faster_rcnn.pth
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bottle_detector.datasets import CocoBottleDataset, get_train_transforms
from bottle_detector.engine import train_model
from bottle_detector.models import build_model, AVAILABLE_MODELS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=AVAILABLE_MODELS, required=True)
    p.add_argument("--data-root", required=True, help="Folder containing train/valid/test subfolders")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", default="checkpoints/model.pth")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_root, "train")
    dataset = CocoBottleDataset(
        root=train_dir,
        annotation=os.path.join(train_dir, "_annotations.coco.json"),
        transforms=get_train_transforms(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CocoBottleDataset.collate_fn,
    )

    model = build_model(args.model, num_classes=2, pretrained=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    train_model(model, data_loader, optimizer, device, num_epochs=args.epochs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
