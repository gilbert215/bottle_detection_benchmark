"""
COCO-format dataset loader for the bottle detection task.

Refactored from the original training notebook. Supports any export in
COCO JSON format (e.g. Roboflow's "COCO" export option), with a single
foreground class ("bottle").
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoBottleDataset(Dataset):
    """Loads images + bounding boxes from a COCO-format annotation file.

    Args:
        root: Directory containing the images for this split.
        annotation: Path to the COCO-format ``_annotations.coco.json`` file.
        transforms: Optional callable applied to the PIL image. Should return
            a tensor (use :mod:`bottle_detector.datasets.transforms`).
    """

    def __init__(self, root: str, annotation: str, transforms: Optional[Callable] = None):
        self.root = root
        self.transforms = transforms

        with open(annotation) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.img_to_anns: dict[int, list] = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        image_id = img_info["id"]

        anns = self.img_to_anns.get(image_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single foreground class: "bottle"

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, image_id

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """Collate function for use with torch DataLoader (variable #boxes per image)."""
        return tuple(zip(*batch))
