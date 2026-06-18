# Bottle Detection Benchmark

Object detection for a single class — **bottles** — comparing three detector
architectures on the same custom dataset: **SSD300-VGG16**, **Faster R-CNN
(ResNet50-FPN)**, and **YOLOv8**.

The project started as a single Kaggle notebook training SSD300-VGG16 on a
COCO-format bottle dataset. This repo restructures that work into a reusable
package and adds ready-to-run training/evaluation code for the other two
architectures, so the comparison can actually be reproduced rather than
just claimed.

## Project structure

```
bottle-detection-benchmark/
├── configs/                  # Hyperparameter configs per model
├── data/                      # Dataset (not committed) + setup instructions
├── notebooks/                 # Original exploratory notebook (SSD)
├── src/bottle_detector/       # Installable package
│   ├── datasets/              # COCO-format dataset + transforms
│   ├── models/                # Model factory (SSD, Faster R-CNN)
│   ├── engine/                # train / evaluate / visualize loops
│   └── utils/                 # Shared metric helpers (IoU)
├── scripts/                   # CLI entry points
│   ├── train.py                # Train SSD or Faster R-CNN
│   ├── evaluate.py             # Evaluate SSD or Faster R-CNN
│   ├── convert_coco_to_yolo.py # COCO -> YOLO format converter
│   └── train_yolov8.py         # Train/val YOLOv8 (Ultralytics)
├── results/                   # Benchmark results + sample predictions
├── requirements.txt
└── LICENSE
```

## Why this dataset / problem

Automated bottle detection is a building block for recycling sorting lines,
retail shelf auditing, and litter-monitoring systems. The dataset here is a
single-class COCO-format export (train/valid/test splits) of bottle images.

## Setup

```bash
git clone <this-repo>
cd bottle-detection-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then populate `data/bottles/` as described in [`data/README.md`](data/README.md).

## Usage

### SSD300-VGG16 / Faster R-CNN

```bash
# Train
python scripts/train.py --model ssd300_vgg16 --data-root data/bottles \
    --epochs 10 --batch-size 4 --lr 0.001 --output checkpoints/ssd.pth

python scripts/train.py --model fasterrcnn_resnet50_fpn --data-root data/bottles \
    --epochs 10 --batch-size 4 --lr 0.001 --output checkpoints/faster_rcnn.pth

# Evaluate
python scripts/evaluate.py --model ssd300_vgg16 --checkpoint checkpoints/ssd.pth \
    --data-root data/bottles --split valid --save-predictions
```

### YOLOv8

```bash
python scripts/convert_coco_to_yolo.py --src data/bottles --dst data/bottles_yolo
python scripts/train_yolov8.py --data data/bottles_yolo/dataset.yaml --model yolov8n.pt --epochs 50
```

### Original notebook

`notebooks/01_ssd_original_experiment.ipynb` is preserved as-is for
provenance — it's the source the `src/bottle_detector` package was refactored
from, and reproduces the SSD numbers below directly in a single notebook.

## Results

**Measured** (SSD300-VGG16, 10 epochs, COCO-pretrained, single GPU):

| Split | mAP | Precision (mAP@50) | Recall (mAR@100) | FPS |
|---|---|---|---|---|
| Validation | 0.356 | 0.97 | 0.47 | 35.6 |
| Test | 0.355 | 0.93 | 0.49 | 37.1 |

Faster R-CNN and YOLOv8 rows are scaffolded with working training code but
**not yet run** — see [`results/benchmark_results.md`](results/benchmark_results.md)
for the full table, the exact commands to fill in real numbers, and notes on
each architecture's known speed/accuracy tradeoffs.

## Possible next steps

- Run the Faster R-CNN and YOLOv8 scripts and fill in `results/benchmark_results.md`.
- Try the heavier augmentation pipeline already stubbed out in
  `get_advanced_train_transforms()` (rotation, color jitter, resized crop).
- Export the best model to ONNX/TorchScript for deployment.

## License

MIT — see [`LICENSE`](LICENSE).
