"""Per-fold `dataset.yaml` documents over a per-video YOLO layout.

`clevis-yolo-prepare --input-format endovis` writes
`images/<split>/<video_id>/<frame>.jpg`, keeping every video in its own
directory. RF-DETR's adapter (`clevis_pseudo_labeler/rfdetr/dataset.py`) accepts
a **list** of image directories per split and namespaces them when it flattens,
so a fold is expressed by naming directories — no second copy of the 5,178
images, seven times over.

The split directory a video sits in (`images/train/...` vs `images/val/...`) is
an artifact of how the dataset was first exported and says nothing about this
fold. Only `build_folds` decides train from val here, which is why every path is
resolved through `video_splits` rather than assumed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def fold_dataset_yaml(
    *,
    train_videos: Sequence[str],
    val_videos: Sequence[str],
    video_splits: Mapping[str, str],
    names: Sequence[str],
    root: str,
) -> dict[str, Any]:
    """Build the dataset.yaml document for one fold.

    `root` is the source dataset's directory and every emitted path is absolute
    under it. This is not cosmetic: the fold yaml is written under the *run's*
    output root, and RF-DETR's adapter resolves a relative split entry against
    the yaml's own directory — so relative paths send it looking for
    `<output_root>/fold0/images/...`, which does not exist.

    Raises `KeyError` when a video is not in the source layout — dropping it
    silently would shrink the fold and flatter its score — and `ValueError` when
    a video appears on both sides, which would defeat leave-2-out entirely.
    """
    overlap = sorted(set(train_videos) & set(val_videos))
    if overlap:
        raise ValueError(f"video(s) in both train and val: {', '.join(overlap)}")

    base = str(root).rstrip("/")

    def _paths(videos: Sequence[str]) -> list[str]:
        resolved = []
        for video in videos:
            if video not in video_splits:
                raise KeyError(video)
            resolved.append(f"{base}/images/{video_splits[video]}/{video}")
        return resolved

    return {
        "train": _paths(train_videos),
        "val": _paths(val_videos),
        "nc": len(names),
        "names": list(names),
    }
