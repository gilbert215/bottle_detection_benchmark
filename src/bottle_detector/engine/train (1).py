"""Training loop shared by SSD300-VGG16 and Faster R-CNN.

This is a direct refactor of the loop in the original notebook
(``notebooks/01_ssd_original_experiment.ipynb``) — the only change is that
it now takes ``model``/``optimizer`` as arguments so it isn't tied to a
single architecture.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> list[float]:
    """Fine-tune a torchvision detection model.

    Returns:
        List of average training loss per epoch.
    """
    model.to(device)
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, targets, _ in data_loader:  # image_id is unused during training
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        if lr_scheduler is not None:
            lr_scheduler.step()

        avg_loss = total_loss / len(data_loader)
        history.append(avg_loss)
        logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, num_epochs, avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return history
