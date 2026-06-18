"""Evaluation loop shared by SSD300-VGG16 and Faster R-CNN.

Refactored from the original notebook's evaluation cell. Computes mean IoU,
COCO-style mAP / mAP@50 / mAR (via ``torchmetrics``), and inference FPS, and
optionally saves annotated prediction images for qualitative inspection.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from bottle_detector.utils.metrics import calculate_iou


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Optional[str] = None,
    save_predictions: bool = False,
) -> dict:
    """Evaluate a detection model on a labeled split.

    Args:
        model: A trained torchvision detection model.
        data_loader: DataLoader yielding (images, targets, image_ids).
        device: torch device to run inference on.
        output_dir: Where to save annotated prediction images (if requested).
        save_predictions: Whether to write annotated images to ``output_dir``.

    Returns:
        Dict with keys: ``mean_iou``, ``mAP``, ``precision`` (mAP@50),
        ``recall`` (mAR@100), ``fps``.
    """
    model.eval()
    model.to(device)

    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metric = MeanAveragePrecision()

    total_iou = 0.0
    num_samples = 0
    total_time = 0.0

    for batch_idx, (images, targets, _) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = [img.to(device) for img in images]

        start = time.time()
        predictions = model(images)
        total_time += time.time() - start

        for i in range(len(images)):
            pred_boxes = predictions[i]["boxes"].cpu().numpy()
            pred_scores = predictions[i]["scores"].cpu().numpy()
            pred_labels = predictions[i]["labels"].cpu().numpy().astype(int)
            gt_boxes = targets[i]["boxes"].cpu().numpy()
            gt_labels = targets[i]["labels"].cpu().numpy().astype(int)

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = [calculate_iou(p, g) for p in pred_boxes for g in gt_boxes]
                total_iou += sum(ious) / len(ious)
                num_samples += 1

            metric.update(
                preds=[
                    {
                        "boxes": torch.tensor(pred_boxes),
                        "scores": torch.tensor(pred_scores),
                        "labels": torch.tensor(pred_labels),
                    }
                ],
                target=[{"boxes": torch.tensor(gt_boxes), "labels": torch.tensor(gt_labels)}],
            )

            if save_predictions and output_dir:
                image_tensor = (images[i] * 255).byte().cpu()
                drawn = draw_bounding_boxes(
                    image_tensor,
                    torch.tensor(pred_boxes, dtype=torch.int),
                    labels=[str(c) for c in pred_labels],
                    colors="red",
                    width=2,
                )
                save_path = os.path.join(output_dir, f"result_{batch_idx}_{i}.png")
                torchvision.utils.save_image(drawn.float() / 255, save_path)

    mean_iou = total_iou / num_samples if num_samples > 0 else 0.0

    metrics = metric.compute()
    mean_ap = metrics["map"].item() if "map" in metrics else 0.0
    precision = metrics["map_50"].item() if "map_50" in metrics else 0.0
    recall = metrics["mar_100"].item() if "mar_100" in metrics else 0.0
    fps = len(data_loader.dataset) / total_time if total_time > 0 else 0.0

    return {
        "mean_iou": mean_iou,
        "mAP": mean_ap,
        "precision": precision,
        "recall": recall,
        "fps": fps,
    }
