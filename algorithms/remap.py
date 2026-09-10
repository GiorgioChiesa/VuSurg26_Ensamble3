"""Re-label a YOLO dataset onto one target class id.

Used to fold the 14-class SurgVU whole-instrument corpus into a wrist-geometry
pretrain: either as a SECOND class alongside the clevis (class 1) or merged into
the SAME class (class 0). Which of those is right is an experiment, not an
assumption, so the class id is a parameter.

The geometry is copied through verbatim — only the class column changes.
"""

from __future__ import annotations


def remap_label_text(text: str, *, class_id: int) -> str:
    """Rewrite every YOLO line to `class_id`, keeping its box untouched."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"expected 5 columns, got {len(parts)}: {line!r}")
        out.append(f"{class_id} {' '.join(parts[1:])}")
    return "".join(f"{line}\n" for line in out)
