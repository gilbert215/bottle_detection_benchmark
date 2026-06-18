#!/usr/bin/env python3
"""
Evaluate a trained SSD300-VGG16 or Faster R-CNN checkpoint on a data split.

Example
-------
    python scripts/evaluate.py --model ssd300_vgg16 --checkpoint checkpoints/ssd.pth \
        --data-root data/bottles --split valid --save-predictions
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bottle_detector.datasets import CocoBottleDataset, get_eval_transforms
from bottle_detector.engine import evaluate_model
from bottle_detector.models import build_model, AVAILABLE_MODELS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=AVAILABLE_MODELS, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--output-dir", default="results/sample_predictions")
    p.add_argument("--results-json", default=None, help="Optional path to dump metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = os.path.join(args.data_root, args.split)
    dataset = CocoBottleDataset(
        root=split_dir,
        annotation=os.path.join(split_dir, "_annotations.coco.json"),
        transforms=get_eval_transforms(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CocoBottleDataset.collate_fn,
    )

    model = build_model(args.model, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results = evaluate_model(
        model,
        data_loader,
        device,
        output_dir=os.path.join(args.output_dir, args.model, args.split),
        save_predictions=args.save_predictions,
    )

    print("-" * 45)
    print(f"{args.model} — {args.split} set")
    print("-" * 45)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    if args.results_json:
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump({"model": args.model, "split": args.split, **results}, f, indent=2)


if __name__ == "__main__":
    main()
