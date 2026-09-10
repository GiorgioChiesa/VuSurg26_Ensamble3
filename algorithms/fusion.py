"""Weighted Box Fusion over several models' detections.

WHY NOT NMS

NMS keeps the single highest-scoring box of a cluster and discards the rest,
which throws away exactly what an ensemble is for: the fact that several models
independently found the same object. WBF instead averages the cluster's
coordinates weighted by score, and scales the result by how many *distinct*
models contributed — so a box three models agree on outranks one that a single
model was confident about.

The per-model bookkeeping matters: one model firing twice on the same object is
not a consensus of two, so contributions are counted by model index, not by box.
"""

from __future__ import annotations

from typing import Any, Sequence


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(
    per_model: Sequence[Sequence[dict[str, Any]]],
    *,
    iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Fuse `per_model[i]` detection lists into one, score-weighted."""
    model_count = len(per_model)
    if model_count == 0:
        return []

    entries = [
        (index, detection)
        for index, detections in enumerate(per_model)
        for detection in detections
    ]
    entries.sort(key=lambda item: float(item[1].get("confidence", 0.0)), reverse=True)

    clusters: list[dict[str, Any]] = []
    for model_index, detection in entries:
        box = [float(v) for v in detection["bbox_xyxy"]]
        score = float(detection.get("confidence", 0.0))
        class_id = detection.get("class_id")
        for cluster in clusters:
            if cluster["class_id"] != class_id:
                continue
            if _iou(cluster["box"], box) < iou_threshold:
                continue
            total = cluster["weight"] + score
            cluster["box"] = [
                (cluster["box"][i] * cluster["weight"] + box[i] * score) / total
                if total > 0
                else cluster["box"][i]
                for i in range(4)
            ]
            cluster["weight"] = total
            cluster["scores"].append(score)
            cluster["models"].add(model_index)
            break
        else:
            clusters.append(
                {
                    "class_id": class_id,
                    "box": box,
                    "weight": score,
                    "scores": [score],
                    "models": {model_index},
                }
            )

    fused = []
    for cluster in clusters:
        mean_score = sum(cluster["scores"]) / len(cluster["scores"])
        agreement = len(cluster["models"]) / model_count
        fused.append(
            {
                "class_id": cluster["class_id"],
                "confidence": mean_score * agreement,
                "bbox_xyxy": cluster["box"],
            }
        )
    return fused
