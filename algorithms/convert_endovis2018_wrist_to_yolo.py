#!/usr/bin/env python3
"""Build a YOLO clevis dataset from EndoVis 2018 part masks.

The box extraction lives in ``clevis_pseudo_labeler.cascade.endovis2018`` and is
unit tested there; this is the I/O around it.

LAYOUT IT READS
    <root>/{train,val}/image/seq_<N>_frame<NNN>.bmp
    <root>/{train,val}/label/seq_<N>_frame<NNN>.bmp   (grayscale class indices)
Image and label share a filename, so the join needs no heuristics.

LAYOUT IT WRITES
    images/<split>/<seq_N>/<frame>.jpg   labels/<split>/<seq_N>/<frame>.txt
One directory per sequence, matching what `clevis-yolo-prepare --input-format
endovis` produces for cat1, so `clevis_pseudo_labeler.cascade.fold_yamls` can
build folds over it without a second layout to support.

Images are SYMLINKED, not copied: the source is a read-only shared mount that
every node already sees, and 3,000 BMPs are not worth duplicating.

Usage:
    python scripts/convert_endovis2018_wrist_to_yolo.py \\
        --source-root /mnt/fast-scratch/shared/datasets/medical/visual/endovis2018 \\
        --output-dir exports/cascade/endovis2018_wrist
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clevis_pseudo_labeler.cascade.endovis2018 import (  # noqa: E402
    CLASS_NAME,
    WRIST_VALUE,
    wrist_boxes,
)

SEQUENCE_RE = re.compile(r"^(seq_\d+)_frame(\d+)$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/fast-scratch/shared/datasets/medical/visual/endovis2018"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--part-value", type=int, default=WRIST_VALUE)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--min-width", type=int, default=2)
    parser.add_argument("--min-height", type=int, default=2)
    parser.add_argument(
        "--keep-negatives",
        action="store_true",
        help="DANGEROUS — export wrist-free frames as empty label files. Off by "
        "default and it should stay off. The wrist is annotated in only 4 of the "
        "18 sequence-splits (seq_3, seq_4, seq_6, seq_10) and there it covers "
        "~100%% of frames (225/225, 225/225, 222/225, 300/300); the other 14 have "
        "ZERO wrist pixels despite showing instruments. Those frames are not "
        "clevis-free scenes, they are scenes annotated under a protocol that does "
        "not separate the wrist, so exporting them as negatives teaches the model "
        "that instruments have no clevis.",
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Put EndoVis's own val split into train too. For a pretrain there is "
        "nothing to hold out here — the real validation is the cat1 k-fold.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    counters: collections.Counter[str] = collections.Counter()
    per_sequence: collections.Counter[str] = collections.Counter()
    splits_seen: set[str] = set()

    for source_split in ("train", "val"):
        label_dir = args.source_root / source_split / "label"
        image_dir = args.source_root / source_split / "image"
        if not label_dir.is_dir():
            continue
        split = "train" if args.all_train else source_split

        for label_path in sorted(label_dir.glob("*.bmp")):
            match = SEQUENCE_RE.match(label_path.stem)
            if not match:
                counters["unparseable_name"] += 1
                continue
            sequence, frame = match.group(1), match.group(2)
            image_path = image_dir / label_path.name
            if not image_path.exists():
                counters["missing_image"] += 1
                continue

            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                counters["unreadable_label"] += 1
                continue

            boxes = wrist_boxes(
                label,
                part_value=args.part_value,
                min_area=args.min_area,
                min_width=args.min_width,
                min_height=args.min_height,
            )
            counters["frames_read"] += 1
            if not boxes and not args.keep_negatives:
                counters["frames_without_wrist_skipped"] += 1
                continue

            splits_seen.add(split)
            counters["frames_written"] += 1
            counters["boxes"] += len(boxes)
            per_sequence[f"{split}/{sequence}"] += 1

            out_label = output / "labels" / split / sequence / f"{frame}.txt"
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)
            )

            # .bmp on purpose: symlinking means the extension has to stay the
            # source's, and the YOLO reader accepts bmp (IMAGE_EXTENSIONS).
            out_image = output / "images" / split / sequence / f"{frame}.bmp"
            out_image.parent.mkdir(parents=True, exist_ok=True)
            if not out_image.exists():
                out_image.symlink_to(image_path.resolve())

    document = {
        **{split: f"images/{split}" for split in sorted(splits_seen)},
        "nc": 1,
        "names": [CLASS_NAME],
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(document, sort_keys=False))

    summary = {
        "format": "clevis.endovis2018-wrist-dataset",
        "version": 1,
        "source_root": str(args.source_root),
        "part_value": args.part_value,
        "filters": {
            "min_area": args.min_area,
            "min_width": args.min_width,
            "min_height": args.min_height,
        },
        "keep_negatives": args.keep_negatives,
        "all_train": args.all_train,
        "counters": dict(counters),
        "per_sequence": dict(sorted(per_sequence.items())),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
