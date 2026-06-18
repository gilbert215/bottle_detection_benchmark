#!/usr/bin/env python3
"""
Convert a COCO-format export (e.g. Roboflow "COCO" export) into the
YOLO/Ultralytics directory + label format, and emit a ``dataset.yaml``.

Expected input layout (one folder per split, each with its own COCO json —
this matches Roboflow's default COCO export):

    data/bottles/
      train/_annotations.coco.json, *.jpg
      valid/_annotations.coco.json, *.jpg
      test/_annotations.coco.json,  *.jpg

Output layout:

    data/bottles_yolo/
      images/{train,valid,test}/*.jpg
      labels/{train,valid,test}/*.txt
      dataset.yaml

Usage
-----
    python scripts/convert_coco_to_yolo.py --src data/bottles --dst data/bottles_yolo
"""
import argparse
import json
import os
import shutil


def convert_split(src_split_dir: str, dst_dir: str, split: str):
    ann_path = os.path.join(src_split_dir, "_annotations.coco.json")
    with open(ann_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out_dir = os.path.join(dst_dir, "images", split)
    lbl_out_dir = os.path.join(dst_dir, "labels", split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for image_id, img_info in images.items():
        file_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        src_img_path = os.path.join(src_split_dir, file_name)
        dst_img_path = os.path.join(img_out_dir, os.path.basename(file_name))
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        lines = []
        for ann in anns_by_image.get(image_id, []):
            x, y, bw, bh = ann["bbox"]
            # YOLO format: class cx cy w h, all normalized to [0, 1]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")  # class 0 = bottle

        label_path = os.path.join(lbl_out_dir, os.path.splitext(os.path.basename(file_name))[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"[{split}] converted {len(images)} images -> {img_out_dir}")


def write_dataset_yaml(dst_dir: str):
    yaml_path = os.path.join(dst_dir, "dataset.yaml")
    content = (
        f"path: {os.path.abspath(dst_dir)}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "names:\n"
        "  0: bottle\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote {yaml_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="COCO-format dataset root (with train/valid/test subfolders)")
    p.add_argument("--dst", required=True, help="Output directory for YOLO-format dataset")
    args = p.parse_args()

    for split in ("train", "valid", "test"):
        split_dir = os.path.join(args.src, split)
        if os.path.isdir(split_dir):
            convert_split(split_dir, args.dst, split)
        else:
            print(f"Skipping missing split: {split_dir}")

    write_dataset_yaml(args.dst)


if __name__ == "__main__":
    main()
