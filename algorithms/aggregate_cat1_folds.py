#!/usr/bin/env python3
"""Score cat1 variants under 7-fold leave-2-out and aggregate the folds.

WHY FOLDS FOR A ZERO-SHOT BRANCH

The patch branch trains nothing on cat1, so "held out" is vacuous for it -- every
video is validation. The folds still matter for two reasons:

  1. The hybrid track (pretrain on SurgVU pseudo-labels, finetune on cat1 GT)
     DOES train on cat1, and its honest number is a fold average. A zero-shot
     number computed over all 7 videos at once is not comparable to that; the
     same fold partition and the same averaging make them comparable.
  2. The sibling repo's reference numbers (claude_traker METHOD.md 25.1:
     YOLOv8n 0.1738, RT-DETR-L 0.2203, best-ever 0.3317) are leave-2-out fold
     averages. Reading an all-7 number against them overstates by an unknown
     amount.

The partition is the rotation "videos (i, i+1) mod 7 are validation for fold i",
so every video lands in validation exactly twice -- matching eval.kfold_val_size:
2 in the sibling repo.

Usage:
    python scripts/aggregate_cat1_folds.py \\
        --config configs/cat1_patch.yaml \\
        --variant patch_sam3=data/cache_cat1_gt/5c_clevis_patch/clevis_patch_boxes__sam3_difference.json \\
        --output-root runs/clevis_patch --run-id 03_sam3_kfold
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("benchmark_endovis_boxes")


# The repo root ahead of site-packages. In the runtime container the INSTALLED
# clevis_pseudo_labeler is older than this checkout and has no `patch_localizer`,
# so without this the import below dies with ModuleNotFoundError even though the
# module is right there in the bind-mounted tree. Prepended, not appended: an
# installed copy that merely lags would otherwise shadow the checkout silently,
# which is the worse failure -- it scores with a stale fold rotation instead of
# refusing to run.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported, never re-implemented: the trainer holds out videos using this same
# partition, and a second copy of the rotation would drift without failing --
# the model would simply be scored on frames it had trained on.
from clevis_pseudo_labeler.patch_localizer.folds import build_folds  # noqa: E402


def score_subset(
    gt_records: list,
    pred_records: list,
    *,
    video_ids: set[str],
    thresholds: tuple[float, ...],
    match_iou: float,
    eval_scope: str,
    class_aware: bool,
) -> dict[str, Any]:
    """Score exactly the frames of these videos.

    Both sides are filtered by video, never only the GT: leaving predictions from
    other videos in would let them be scored as false positives on frames they
    could never match, which silently penalises whichever fold happens to hold
    the busiest videos.
    """
    gt_subset = [r for r in gt_records if str(r.image_key[0]) in video_ids]
    pred_subset = [r for r in pred_records if str(r.image_key[0]) in video_ids]
    if not gt_subset:
        return {}
    summary, _, _ = bench.evaluate(
        gt_subset,
        pred_subset,
        iou_thresholds=thresholds,
        match_threshold=match_iou,
        scale_predictions=True,
        eval_scope=eval_scope,
        class_aware=class_aware,
    )
    per_class = bench.evaluate_per_class(
        gt_subset,
        pred_subset,
        iou_thresholds=thresholds,
        match_threshold=match_iou,
        scale_predictions=True,
        eval_scope=eval_scope,
    )
    return {
        "gt_boxes": len(gt_subset),
        "pred_boxes": len(pred_subset),
        "map_50_95": per_class["map_50_95"],
        "ap50": per_class["ap50"],
        "supported_classes": per_class["supported_classes"],
        "precision": summary.get("precision", 0.0),
        "recall": summary.get("recall", 0.0),
        "median_matched_iou": summary.get("median_matched_iou", 0.0),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument(
        "--variant", action="append", required=True, metavar="NAME=PATH", default=[]
    )
    parser.add_argument("--val-size", type=int, default=2)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--iou-thresholds", default="0.50:0.95:0.05")
    parser.add_argument("--output-root", type=Path, default=Path("runs/clevis_patch"))
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from clevis_pseudo_labeler.config import load_config

    cfg = load_config(args.config)
    ev = cfg.clevis_eval
    gt_dir = args.gt or Path(ev.gt_dir)
    category_map = {
        str(name): int(value.id if hasattr(value, "id") else value["id"])
        for name, value in (ev.category_map or {}).items()
        if value is not None
    }
    thresholds = bench._parse_thresholds(args.iou_thresholds)

    gt_records = bench.load_ground_truth(Path(gt_dir))
    if not gt_records:
        raise SystemExit(f"No ground-truth boxes found under {gt_dir}")
    video_ids = sorted({str(r.image_key[0]) for r in gt_records})
    folds = build_folds(video_ids, val_size=args.val_size)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ground truth: {len(gt_records)} boxes over {len(video_ids)} videos")
    print(f"Protocol:     {len(folds)}-fold leave-{args.val_size}-out")
    print(f"              every video appears in validation exactly {args.val_size}x")

    results: list[dict[str, Any]] = []
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name, path = name.strip(), Path(path.strip())
        if not path.exists():
            print(f"  {name:26s} SKIPPED (no artifact at {path})")
            continue
        unmapped: Counter = Counter()
        pred_records = bench.load_predictions(
            [path],
            image_width=ev.pred_width,
            image_height=ev.pred_height,
            category_map=category_map or None,
            unmapped_classes=unmapped,
        )

        fold_rows = []
        for fold in folds:
            row = score_subset(
                gt_records,
                pred_records,
                video_ids=set(fold["val_videos"]),
                thresholds=thresholds,
                match_iou=args.match_iou,
                eval_scope=ev.eval_scope,
                class_aware=ev.class_aware,
            )
            if row:
                fold_rows.append({**fold, **row})

        all7 = score_subset(
            gt_records,
            pred_records,
            video_ids=set(video_ids),
            thresholds=thresholds,
            match_iou=args.match_iou,
            eval_scope=ev.eval_scope,
            class_aware=ev.class_aware,
        )
        maps = [r["map_50_95"] for r in fold_rows]
        ap50s = [r["ap50"] for r in fold_rows]
        results.append(
            {
                "variant": name,
                "path": str(path),
                "folds": fold_rows,
                "fold_mean_map_50_95": statistics.fmean(maps) if maps else 0.0,
                "fold_stdev_map_50_95": statistics.pstdev(maps) if len(maps) > 1 else 0.0,
                "fold_mean_ap50": statistics.fmean(ap50s) if ap50s else 0.0,
                "all_videos": all7,
                "unmapped_prediction_classes": dict(unmapped),
            }
        )

    print()
    header = (
        f"{'variant':26s} {'foldMAP':>8s} {'sd':>6s} {'foldAP50':>9s} "
        f"{'all7MAP':>8s} {'all7AP50':>9s}"
    )
    print("7-fold leave-2-out (foldMAP is the figure comparable to the sibling repo)")
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['variant']:26s} {result['fold_mean_map_50_95']:8.4f} "
            f"{result['fold_stdev_map_50_95']:6.4f} {result['fold_mean_ap50']:9.4f} "
            f"{result['all_videos'].get('map_50_95', 0.0):8.4f} "
            f"{result['all_videos'].get('ap50', 0.0):9.4f}"
        )
    print()
    print("Reference (claude_traker METHOD.md 25.1, same leave-2-out protocol):")
    print("  YOLOv8n 0.1738 | RT-DETR-L 0.2203 | best ever (terzetto+TTA) 0.3317")

    print()
    print("per-fold mAP:")
    for result in results:
        cells = " ".join(f"{r['map_50_95']:.4f}" for r in result["folds"])
        print(f"  {result['variant']:26s} {cells}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_dir": str(gt_dir),
        "protocol": {
            "folds": len(folds),
            "val_size": args.val_size,
            "partition": folds,
        },
        "settings": {
            "eval_scope": ev.eval_scope,
            "class_aware": ev.class_aware,
            "match_iou": args.match_iou,
            "pred_width": ev.pred_width,
            "pred_height": ev.pred_height,
        },
        "variants": results,
    }
    report_path = out_dir / "kfold_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print()
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
