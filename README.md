# Bottle Detection Benchmark

Object detection for a single class, **bottles**, comparing three detector architectures on the same custom dataset: **SSD300-VGG16**, **Faster R-CNN (ResNet50-FPN)**, and **YOLOv8**.

The project began as a single Kaggle notebook training SSD300-VGG16 on a COCO-format bottle dataset. This repository restructures that work into a reusable package and extends it with training and evaluation pipelines for Faster R-CNN and YOLOv8, enabling reproducible comparisons across different object detection architectures.

## Project Structure

```text
bottle-detection-benchmark/
├── configs/                   # Hyperparameter configs per model
├── data/                      # Dataset (not committed) + setup instructions
├── notebooks/                 # Original exploratory notebook (SSD)
├── src/bottle_detector/       # Installable package
│   ├── datasets/              # COCO-format dataset + transforms
│   ├── models/                # Model factory (SSD, Faster R-CNN)
│   ├── engine/                # Train, evaluate, visualize loops
│   └── utils/                 # Shared metric helpers (IoU)
├── scripts/                   # CLI entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── convert_coco_to_yolo.py
│   └── train_yolov8.py
├── results/                   # Benchmark results and sample predictions
├── requirements.txt
└── LICENSE
```

## Motivation

Automated bottle detection is useful for recycling sorting systems, retail shelf monitoring, and environmental litter detection. The dataset used in this project is a single-class COCO-format dataset with train, validation, and test splits.

## Setup

```bash
git clone <this-repo>
cd bottle-detection-benchmark

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Populate `data/bottles/` according to the instructions in `data/README.md`.

## Usage

### SSD300-VGG16 and Faster R-CNN

#### Train SSD300-VGG16

```bash
python scripts/train.py \
    --model ssd300_vgg16 \
    --data-root data/bottles \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.001 \
    --output checkpoints/ssd.pth
```

#### Train Faster R-CNN

```bash
python scripts/train.py \
    --model fasterrcnn_resnet50_fpn \
    --data-root data/bottles \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.001 \
    --output checkpoints/faster_rcnn.pth
```

#### Evaluate

```bash
python scripts/evaluate.py \
    --model ssd300_vgg16 \
    --checkpoint checkpoints/ssd.pth \
    --data-root data/bottles \
    --split valid \
    --save-predictions
```

### YOLOv8

```bash
python scripts/convert_coco_to_yolo.py \
    --src data/bottles \
    --dst data/bottles_yolo

python scripts/train_yolov8.py \
    --data data/bottles_yolo/dataset.yaml \
    --model yolov8n.pt \
    --epochs 50
```

### Original Notebook

`notebooks/01_ssd_original_experiment.ipynb` contains the initial SSD experiment from which the package was refactored.

## Results

All models were initialized from COCO pretrained weights and evaluated on the same validation and test sets.

### Validation Performance

| Model                     | mAP@50:95 | Precision | Recall |  FPS |
| ------------------------- | :-------: | :-------: | :----: | :--: |
| SSD300-VGG16              |   0.356   |    0.97   |  0.47  | 35.6 |
| Faster R-CNN ResNet50-FPN |   0.441   |    0.94   |  0.66  | 12.8 |
| YOLOv8n                   |   0.425   |    0.95   |  0.63  | 78.4 |

### Test Performance

| Model                     | mAP@50:95 | Precision | Recall |  FPS |
| ------------------------- | :-------: | :-------: | :----: | :--: |
| SSD300-VGG16              |   0.355   |    0.93   |  0.49  | 37.1 |
| Faster R-CNN ResNet50-FPN |   0.438   |    0.93   |  0.65  | 12.5 |
| YOLOv8n                   |   0.421   |    0.94   |  0.62  | 80.2 |

## Key Findings

* **Faster R-CNN (ResNet50-FPN)** achieved the highest detection accuracy with a test mAP@50:95 of **0.438**. Its two-stage architecture provided stronger localization performance, although inference speed was lower.

* **YOLOv8n** offered the best balance between speed and accuracy, achieving a test mAP@50:95 of **0.421** while exceeding **80 FPS**, making it suitable for real-time deployment.

* **SSD300-VGG16** served as a lightweight baseline. Although it provided competitive inference speed, its recall and overall detection performance were lower than the other architectures.

## Benchmark Ranking

| Rank | Model                     | Main Strength                |
| :--: | ------------------------- | ---------------------------- |
|  1  | YOLOv8n                   | Best speed-accuracy tradeoff |
|  2  | Faster R-CNN ResNet50-FPN | Highest accuracy             |
|  3  | SSD300-VGG16              | Lightweight baseline         |

## Future Work

* Evaluate larger YOLOv8 variants such as `YOLOv8s` and `YOLOv8m`.
* Introduce stronger data augmentation including random rotation, color jitter, and mosaic augmentation.
* Perform hyperparameter optimization.
* Export the best model to ONNX or TorchScript for deployment on edge devices.
* Extend the dataset to multiple recyclable object categories.

## License

This project is released under the MIT License. See `LICENSE` for details.
