"""Shared, framework-agnostic metric helpers."""


def calculate_iou(pred_box, gt_box) -> float:
    """Intersection-over-Union for two ``[x1, y1, x2, y2]`` boxes."""
    inter_xmin = max(pred_box[0], gt_box[0])
    inter_ymin = max(pred_box[1], gt_box[1])
    inter_xmax = min(pred_box[2], gt_box[2])
    inter_ymax = min(pred_box[3], gt_box[3])

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    union_area = pred_area + gt_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0
