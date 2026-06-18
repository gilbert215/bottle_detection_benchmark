"""
Model factory for the torchvision-based detectors used in this project.

Both SSD300-VGG16 and Faster R-CNN (ResNet50-FPN) expose the same
``model(images, targets) -> loss_dict`` / ``model(images) -> predictions``
API in torchvision, so a single training/eval loop (see
``bottle_detector.engine``) works for either one unmodified.

YOLOv8 is intentionally NOT included here — Ultralytics' YOLO trains and
loads data through its own API/format, so it's handled separately by
``scripts/train_yolov8.py``.
"""
from __future__ import annotations

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

AVAILABLE_MODELS = ("ssd300_vgg16", "fasterrcnn_resnet50_fpn")


def build_model(name: str, num_classes: int = 2, pretrained: bool = True):
    """Build a torchvision detection model for fine-tuning.

    Args:
        name: One of ``AVAILABLE_MODELS``.
        num_classes: Number of classes including background
            (e.g. 2 = background + bottle).
        pretrained: Whether to start from COCO-pretrained weights.

    Returns:
        An ``nn.Module`` ready for training.
    """
    if name == "ssd300_vgg16":
        weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model = ssd300_vgg16(weights=weights)
        # Matches the original notebook's approach to adapting the head.
        model.head.classification_head.num_classes = num_classes
        return model

    if name == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model '{name}'. Choose from {AVAILABLE_MODELS}.")
