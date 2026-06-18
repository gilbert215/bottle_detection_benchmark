# Dataset

This project expects a single-class ("bottle") object detection dataset
exported in **COCO JSON** format, with the following layout:

```
data/bottles/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

This matches the default "COCO" export option from [Roboflow](https://roboflow.com/),
which is what the original experiment (`notebooks/01_ssd_original_experiment.ipynb`)
was trained on.

## Getting the data

1. Export your Roboflow project (or any annotation tool that supports COCO
   export) in COCO format and unzip it into `data/bottles/`.
2. For the SSD / Faster R-CNN scripts, no further conversion is needed —
   they read the COCO JSON directly via `CocoBottleDataset`.
3. For YOLOv8, convert the COCO export to YOLO format first:

   ```bash
   python scripts/convert_coco_to_yolo.py --src data/bottles --dst data/bottles_yolo
   ```

The actual image/annotation files are **not** committed to this repo (see
`.gitignore`) — only the code and folder structure are tracked.
