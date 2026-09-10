#!/usr/bin/env python3
"""Materialise a YOLO dataset re-labelled onto one class id.

Images are SYMLINKED: only the label text changes, so a second copy of the
pixels buys nothing and costs minutes on a GlusterFS mount.

Usage:
    python scripts/remap_yolo_class.py \
        --input exports/cascade/auto_labeling-087-ocr \
        --output-dir exports/cascade/surgvu_as_class1 \
        --class-id 1 --class-name whole_instrument
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from clevis_pseudo_labeler.cascade.remap import remap_label_text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--names", default="", help="Comma-separated full name list for nc.")
    args = parser.parse_args()

    names = [n for n in args.names.split(",") if n] or [args.class_name]
    counts = {"images": 0, "labels": 0, "boxes": 0}
    splits = []

    for split_dir in sorted((args.input / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        splits.append(split)
        for image in sorted(split_dir.rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = image.relative_to(args.input / "images")
            out_image = args.output_dir / "images" / rel
            out_image.parent.mkdir(parents=True, exist_ok=True)
            if not out_image.exists():
                out_image.symlink_to(image.resolve())
            counts["images"] += 1

            source_label = args.input / "labels" / rel.with_suffix(".txt")
            out_label = args.output_dir / "labels" / rel.with_suffix(".txt")
            out_label.parent.mkdir(parents=True, exist_ok=True)
            text = source_label.read_text() if source_label.exists() else ""
            remapped = remap_label_text(text, class_id=args.class_id)
            out_label.write_text(remapped)
            counts["labels"] += 1
            counts["boxes"] += len(remapped.splitlines())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(
            {**{s: f"images/{s}" for s in splits}, "nc": len(names), "names": names},
            sort_keys=False,
        )
    )
    print(json.dumps({"source": str(args.input), "class_id": args.class_id, **counts}, indent=2))


if __name__ == "__main__":
    main()
