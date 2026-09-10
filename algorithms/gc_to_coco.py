#!/usr/bin/env python3
"""Grand Challenge `surgical-tools.json` -> COCO, for the overlay renderer.

The submission writes the GC shape (`{"boxes": [{"name": "slice_nr_7_needle_driver",
"corners": [...], "probability": ...}]}`); every scoring and rendering tool in
this repo reads COCO. The frame index and the class both live inside `name`, so
the conversion is a parse, not a lookup.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CATEGORIES = [
    "grasping retractor", "cadiere forceps", "bipolar forceps", "force bipolar",
    "clip applier", "stapler", "permanent cautery hook spatula",
    "monopolar curved scissor", "vessel sealer", "tip up fenestrated grasper",
    "bipolar dissector", "needle driver", "prograsp forceps", "suction irrigator",
]
ID_OF = {name.replace(" ", "_"): index for index, name in enumerate(CATEGORIES)}
NAME_RE = re.compile(r"^slice_nr_(\d+)_(.+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    images: dict[int, dict] = {}
    annotations = []
    unknown: set[str] = set()
    for box in json.loads(args.input.read_text())["boxes"]:
        match = NAME_RE.match(box["name"])
        if not match:
            continue
        frame, tool = int(match.group(1)), match.group(2)
        if tool not in ID_OF:
            unknown.add(tool)
            continue
        xs = [c[0] for c in box["corners"]]
        ys = [c[1] for c in box["corners"]]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        images.setdefault(frame, {
            "id": frame, "video_id": args.video_id,
            "width": args.width, "height": args.height,
            "file_name": f"{frame:010d}.jpg",
        })
        annotations.append({
            "id": len(annotations) + 1, "image_id": frame,
            "category_id": ID_OF[tool], "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            "score": float(box.get("probability", 1.0)),
        })

    args.output.write_text(json.dumps({
        "video_id": args.video_id,
        "images": [images[k] for k in sorted(images)],
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CATEGORIES)],
    }))
    if unknown:
        print(f"  classi ignote saltate: {sorted(unknown)}")
    print(f"{len(annotations)} box su {len(images)} frame -> {args.output}")


if __name__ == "__main__":
    main()
