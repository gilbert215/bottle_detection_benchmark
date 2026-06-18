"""Qualitative visualization: ground-truth vs. predicted boxes."""
from __future__ import annotations

import torchvision.transforms.functional as F
from PIL import ImageDraw

from bottle_detector.utils.metrics import calculate_iou


def visualize_predictions(model, data_loader, device, num_images: int = 5, threshold: float = 0.5):
    """Draw ground-truth boxes (green) and predictions (red) and show them inline.

    Intended for notebook/interactive use. For headless saving of prediction
    images, use ``evaluate_model(..., save_predictions=True)`` instead.
    """
    import matplotlib.pyplot as plt
    import torch

    model.eval()
    model.to(device)

    images_shown = 0
    for images, targets, _ in data_loader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for i in range(len(images)):
            if images_shown >= num_images:
                return

            img = F.to_pil_image(images[i].cpu())
            draw = ImageDraw.Draw(img)

            gt_boxes = targets[i]["boxes"].cpu().numpy()
            gt_labels = targets[i]["labels"].cpu().numpy()
            for box, label in zip(gt_boxes, gt_labels):
                draw.rectangle(list(box), outline="green", width=3)
                draw.text((box[0], box[1] - 10), f"GT: {label}", fill="green")

            pred_boxes = outputs[i]["boxes"].cpu().numpy()
            pred_scores = outputs[i]["scores"].cpu().numpy()
            pred_labels = outputs[i]["labels"].cpu().numpy()

            correct, total = 0, 0
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < threshold:
                    continue
                draw.rectangle(list(box), outline="red", width=3)
                draw.text((box[0], box[1] - 10), f"P: {label} ({score:.2f})", fill="red")
                if any(calculate_iou(box, gt) > 0.5 for gt in gt_boxes):
                    correct += 1
                total += 1

            accuracy = (correct / total) * 100 if total > 0 else 0.0
            draw.text((10, 10), f"Accuracy: {accuracy:.2f}%", fill="blue")

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.show()

            images_shown += 1
