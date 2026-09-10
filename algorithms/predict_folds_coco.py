#!/usr/bin/env python3
"""Run each fold's checkpoint over its own validation videos, as COCO.

WHY PER FOLD AND NOT ONE FILE

Under leave-2-out every video is validation in exactly two folds, so a single
merged prediction file cannot serve both without at least one of them scoring a
model that trained on that video. Each fold therefore gets its own directory
holding only its val videos, and `scripts/aggregate_cat1_folds.py` is passed all
of them as separate `--variant`s; the fold-f row of variant f is the honest one.

Usage:
    python scripts/predict_folds_coco.py \\
        --kfold-root runs/cascade_kfold/clevis14 \\
        --dataset exports/cascade/cat1_clevis_yolo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from clevis_pseudo_labeler.cascade.coco_predictions import (
    coco_document,
    frame_index_from_stem,
)
from clevis_pseudo_labeler.rfdetr.roster_annotate import (
    _load_rfdetr_model,
    detections_from_rfdetr,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kfold-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint-name", default="checkpoint_best_ema.pth")
    # 0.05 mirrors the patch branch: in COCO AP a low-scoring false positive
    # ranked below every true positive costs almost nothing, while a missed box
    # costs a whole GT.
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def _val_videos(fold_dir: Path) -> list[tuple[str, Path]]:
    document = yaml.safe_load((fold_dir / "dataset.yaml").read_text())
    return [(Path(p).name, Path(p)) for p in document["val"]]


def main() -> None:
    args = build_arg_parser().parse_args()
    names = yaml.safe_load((args.dataset / "dataset.yaml").read_text())["names"]

    for fold_dir in sorted(args.kfold_root.glob("fold*")):
        checkpoint = fold_dir / "run" / args.checkpoint_name
        if not checkpoint.exists():
            print(f"{fold_dir.name}: nessun {args.checkpoint_name}, salto")
            continue
        detector = _load_rfdetr_model(checkpoint)
        output_dir = fold_dir / "preds"
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_id, image_dir in _val_videos(fold_dir):
            frames: list[tuple[int, list[dict]]] = []
            paths = sorted(image_dir.glob("*.jpg"))
            width = height = None
            for start in range(0, len(paths), args.batch_size):
                chunk = paths[start : start + args.batch_size]
                images = []
                for path in chunk:
                    bgr = cv2.imread(str(path))
                    if height is None:
                        height, width = bgr.shape[:2]
                    images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                raw = detector.predict(images, threshold=args.conf, include_source_image=False)
                if not isinstance(raw, (list, tuple)):
                    raw = [raw]
                for path, prediction in zip(chunk, raw):
                    frames.append(
                        (
                            frame_index_from_stem(path.stem),
                            detections_from_rfdetr(prediction, class_names=names),
                        )
                    )

            document = coco_document(
                video_id=video_id,
                width=width or 640,
                height=height or 512,
                frames=frames,
                categories=[{"id": i, "name": n} for i, n in enumerate(names)],
            )
            # `coco_*.json` because _expand_prediction_paths globs that prefix.
            destination = output_dir / f"coco_{video_id}.json"
            destination.write_text(json.dumps(document))
            print(
                f"{fold_dir.name}/{video_id}: {len(document['images'])} frame, "
                f"{len(document['annotations'])} box -> {destination}",
                flush=True,
            )


if __name__ == "__main__":
    main()
