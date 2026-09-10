#!/usr/bin/env python3
"""Ensemble several k-fold runs per fold, fused with WBF, written as COCO.

Every run passed in must share the SAME fold partition, so the models ensembled
for fold f are all models that trained on fold f's train videos and none that
saw its val videos. That is what keeps the ensemble honest.

Usage:
    python scripts/predict_folds_ensemble.py \\
        --kfold-root runs/cascade_kfold/clevis14 \\
        --kfold-root runs/cascade_kfold/clevis14_ev18 \\
        --kfold-root runs/cascade_kfold/clevis14_sarrarp \\
        --dataset exports/cascade/cat1_clevis_yolo \\
        --output-root runs/cascade_kfold/ensemble3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

# The repo root ahead of site-packages. The runtime image's INSTALLED
# clevis_pseudo_labeler is older than this checkout and has no `cascade` package,
# so unless the caller happens to overlay the checkout onto /app this dies with
# ModuleNotFoundError while the module sits right there in the bind-mounted tree.
# Prepended, not appended: an installed copy that merely lags would otherwise
# shadow the checkout silently and predict with stale fusion/TTA code.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clevis_pseudo_labeler.cascade.coco_predictions import (  # noqa: E402
    coco_document,
    frame_index_from_stem,
)
from clevis_pseudo_labeler.cascade.fusion import weighted_box_fusion  # noqa: E402
from clevis_pseudo_labeler.cascade.tta import unflip_horizontal  # noqa: E402
from clevis_pseudo_labeler.rfdetr.roster_annotate import (  # noqa: E402
    _load_rfdetr_model,
    detections_from_rfdetr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kfold-root", action="append", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint-name", default="checkpoint_best_ema.pth")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--flip-tta",
        action="store_true",
        help="Also predict on the mirrored frame and fuse both. Horizontal only: "
        "a surgical scene has no canonical left/right, but it does have a fixed "
        "up/down, so a vertical flip would ask about a scene that never occurs.",
    )
    args = parser.parse_args()

    names = yaml.safe_load((args.dataset / "dataset.yaml").read_text())["names"]

    for fold in range(7):
        checkpoints = [
            root / f"fold{fold}" / "run" / args.checkpoint_name for root in args.kfold_root
        ]
        checkpoints = [c for c in checkpoints if c.exists()]
        if not checkpoints:
            continue
        reference = args.kfold_root[0] / f"fold{fold}" / "dataset.yaml"
        val_dirs = [Path(p) for p in yaml.safe_load(reference.read_text())["val"]]

        models = [_load_rfdetr_model(c) for c in checkpoints]
        output_dir = args.output_root / f"fold{fold}" / "preds"
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_dir in val_dirs:
            video_id = image_dir.name
            # png as well as jpg: the grayscale export is written lossless, so a
            # synthetic channel is never blurred by a re-encode. The RGB export
            # holds no PNGs, so this is a no-op for every run made so far.
            paths = sorted(
                path
                for pattern in ("*.jpg", "*.png")
                for path in image_dir.glob(pattern)
            )
            frames = []
            width = height = None
            for start in range(0, len(paths), args.batch_size):
                chunk = paths[start : start + args.batch_size]
                images = []
                for path in chunk:
                    bgr = cv2.imread(str(path))
                    if height is None:
                        height, width = bgr.shape[:2]
                    images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                per_model_batches = []
                for model in models:
                    raw = model.predict(images, threshold=args.conf, include_source_image=False)
                    if not isinstance(raw, (list, tuple)):
                        raw = [raw]
                    per_model_batches.append(
                        [detections_from_rfdetr(p, class_names=names) for p in raw]
                    )
                    if args.flip_tta:
                        mirrored = [image[:, ::-1].copy() for image in images]
                        raw_flip = model.predict(
                            mirrored, threshold=args.conf, include_source_image=False
                        )
                        if not isinstance(raw_flip, (list, tuple)):
                            raw_flip = [raw_flip]
                        # Each TTA view is a separate ENSEMBLE MEMBER, so a box
                        # both views agree on gets the agreement bonus WBF gives.
                        per_model_batches.append(
                            [
                                unflip_horizontal(
                                    detections_from_rfdetr(p, class_names=names),
                                    width=images[0].shape[1],
                                )
                                for p in raw_flip
                            ]
                        )
                for offset, path in enumerate(chunk):
                    frames.append(
                        (
                            frame_index_from_stem(path.stem),
                            weighted_box_fusion(
                                [batch[offset] for batch in per_model_batches],
                                iou_threshold=args.iou_threshold,
                            ),
                        )
                    )

            document = coco_document(
                video_id=video_id,
                width=width or 640,
                height=height or 512,
                frames=frames,
                categories=[{"id": i, "name": n} for i, n in enumerate(names)],
            )
            destination = output_dir / f"coco_{video_id}.json"
            destination.write_text(json.dumps(document))
            print(
                f"fold{fold}/{video_id}: {len(models)} modelli, "
                f"{len(document['annotations'])} box fusi -> {destination}",
                flush=True,
            )


if __name__ == "__main__":
    main()
