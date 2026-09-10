"""Sesta sottomissione — ensemble RF-DETR-Medium a 3 membri sul clevis cat1.

I box predetti sono il CLEVIS (polso/cerniera dello strumento), che e' cio' che
annota la ground truth cat1 — non lo strumento intero. Eccezione della challenge:
per monopolar curved scissors, tip-up fenestrated grasper, stapler, grasping
retractor e suction irrigator il box di GT racchiude anche le punte, e il corpus
di training e' stato corretto di conseguenza con due giri di review umana.

Tre membri, ognuno con un pretrain esterno diverso, fusi con Weighted Box Fusion
piu' flip TTA: 2 view per membro, 6 forward pass per frame. La scelta di tre
membri e' misurata su tutti i 63 sottoinsiemi dei sei allenati — sopra i tre la
curva e' piatta entro il rumore. Diagonale 7-fold leave-2-out: 0.6414.

Vedi model/README.md per i numeri completi e per il filtro post-fusione.
"""

from pathlib import Path
import json
import os
import sys

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path(os.environ.get("CLEVIS_MODEL_DIR", "/opt/ml/model"))

# cat1 category order — the class ids the models were trained with. Index IS the
# class id, so this list must not be reordered.
CATEGORIES = [
    "grasping retractor",
    "cadiere forceps",
    "bipolar forceps",
    "force bipolar",
    "clip applier",
    "stapler",
    "permanent cautery hook spatula",
    "monopolar curved scissor",
    "vessel sealer",
    "tip up fenestrated grasper",
    "bipolar dissector",
    "needle driver",
    "prograsp forceps",
    "suction irrigator",
]

# Grand Challenge names the tools with underscores. The mapping is mechanical,
# but it is written out rather than computed so a future rename cannot silently
# produce a name the evaluator does not recognise.
GC_NAME = {name: name.replace(" ", "_") for name in CATEGORIES}

# Detections below this score are dropped. 0.05 mirrors the k-fold protocol: in
# COCO AP a low-scoring false positive ranked below every true positive costs
# almost nothing, while a missed box costs a whole ground-truth object.
# Per-model gate, BEFORE fusion. Deliberately low: fusion needs the weak
# agreeing views to see, and what reaches the output is governed by MIN_SCORE.
SCORE_THRESHOLD = float(os.environ.get("CLEVIS_SCORE_THRESHOLD", "0.05"))
# Floor on the score actually REPORTED, applied AFTER fusion. The first
# submission had no such floor and emitted 56,445 boxes over 1,800 frames
# (31.4/frame, median score 0.016), because WBF scales a cluster by the fraction
# of views that found it and a 1-of-8 box leaves fusion at ~0.006.
# 0.3, not 0.5: measured on the first submission's own output, 0.5 leaves 1.18
# boxes/frame against the ground truth's 2.19 — it would discard roughly half
# the real objects. 0.3 leaves 1.88/frame, the closest match to the GT density.
MIN_SCORE = float(os.environ.get("CLEVIS_MIN_SCORE", "0.3"))
# The cat1 ground truth never holds more than 3 boxes in a frame (1: 798 frames,
# 2: 2,615, 3: 1,765) — three instrument arms plus the endoscope.
MAX_PER_FRAME = int(os.environ.get("CLEVIS_MAX_PER_FRAME", "3"))
# WBF fuses within a class only, so a detector hedging between classes reports
# the same object twice: 26,905 cross-class pairs overlapped at IoU > 0.9.
CROSS_CLASS_IOU = float(os.environ.get("CLEVIS_CROSS_CLASS_IOU", "0.55"))
IOU_THRESHOLD = float(os.environ.get("CLEVIS_IOU_THRESHOLD", "0.55"))
BATCH_SIZE = int(os.environ.get("CLEVIS_BATCH_SIZE", "8"))
FLIP_TTA = os.environ.get("CLEVIS_FLIP_TTA", "1") != "0"


# ── fusion (vendored: the container must not depend on the research repo) ────


def _iou(a, b):
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(per_view, *, iou_threshold=IOU_THRESHOLD):
    """Fuse detections from several views, weighting by score and agreement.

    NMS would keep one box per cluster and throw away the agreement between
    views, which is the only reason an ensemble is worth its cost. Contributions
    are counted per view index, so one view firing twice is not a consensus.
    """
    view_count = len(per_view)
    if view_count == 0:
        return []
    entries = [(i, d) for i, view in enumerate(per_view) for d in view]
    entries.sort(key=lambda item: item[1]["score"], reverse=True)

    clusters = []
    for view_index, detection in entries:
        box, score, cls = detection["box"], detection["score"], detection["class_id"]
        for cluster in clusters:
            if cluster["class_id"] != cls or _iou(cluster["box"], box) < iou_threshold:
                continue
            total = cluster["weight"] + score
            cluster["box"] = [
                (cluster["box"][i] * cluster["weight"] + box[i] * score) / total
                if total > 0 else cluster["box"][i]
                for i in range(4)
            ]
            cluster["weight"] = total
            cluster["scores"].append(score)
            cluster["views"].add(view_index)
            break
        else:
            clusters.append({"class_id": cls, "box": list(box), "weight": score,
                             "scores": [score], "views": {view_index}})

    return [
        {
            "class_id": c["class_id"],
            "box": c["box"],
            "score": (sum(c["scores"]) / len(c["scores"])) * (len(c["views"]) / view_count),
        }
        for c in clusters
    ]


def filter_frame(detections):
    """Score floor, cross-class suppression, physical cap — in that order.

    Suppressing before the floor would let a box the floor was about to drop
    take a slot away from a better one.
    """
    survivors = sorted(
        (d for d in detections if d["score"] >= MIN_SCORE),
        key=lambda d: d["score"], reverse=True,
    )
    kept = []
    for detection in survivors:
        if any(_iou(detection["box"], k["box"]) >= CROSS_CLASS_IOU for k in kept):
            continue
        kept.append(detection)
        if len(kept) >= MAX_PER_FRAME:
            break
    return kept


def unflip_horizontal(detections, width):
    """Map boxes predicted on a mirrored frame back onto the original.

    Mirroring swaps which edge is left, so x1 and x2 exchange roles.
    """
    return [
        {**d, "box": [width - d["box"][2], d["box"][1], width - d["box"][0], d["box"][3]]}
        for d in detections
    ]


# ── models ──────────────────────────────────────────────────────────────────


def load_models():
    """Load every checkpoint found under the model directory."""
    import torch
    from rfdetr import from_checkpoint

    # ONLY model/ensemble. The tarball also ships model/whole_instrument, which
    # holds detectors of the WHOLE instrument — a different object entirely. An
    # rglob over the whole tree would silently enrol them as ensemble members and
    # flood every frame with instrument-sized boxes labelled as clevises.
    ensemble_dir = MODEL_PATH / "ensemble"
    root = ensemble_dir if ensemble_dir.is_dir() else MODEL_PATH
    checkpoints = sorted(root.glob("*.pth")) + sorted(root.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"no model checkpoint under {MODEL_PATH}. Upload the model tarball to "
            "Grand Challenge (Algorithm > Models), or bake the weights into "
            "resources/ and set CLEVIS_MODEL_DIR."
        )
    print(f"loading {len(checkpoints)} checkpoint(s) from {MODEL_PATH}", flush=True)
    models = []
    for path in checkpoints:
        print(f"  {path.name}", flush=True)
        # No trust_checkpoint= here: this rfdetr forwards unknown kwargs into
        # RFDETRMediumConfig, which rejects them ("Unknown parameter(s):
        # 'trust_checkpoint'"). The weights are ours, so nothing needs trusting.
        model = from_checkpoint(str(path))
        # FP16 inference. The container logs "Model is not optimized for
        # inference" for every model without this, and the ensemble runs 8
        # forward passes per frame, so the cost is multiplied by eight.
        # compile=False on purpose: JIT-tracing pins the batch size, and the last
        # batch of a video is almost never full — it would need padding, which is
        # a correctness risk for a speed-up we do not need.
        try:
            model.optimize_for_inference(compile=False, dtype=torch.float16)
        except Exception as exc:  # never let an optimisation stop the submission
            print(f"    optimize_for_inference non riuscita ({exc}); si prosegue in fp32", flush=True)
        models.append(model)
    return models


def detections_of(prediction):
    """supervision.Detections -> plain dicts, in xyxy pixel coordinates."""
    out = []
    xyxy = getattr(prediction, "xyxy", None)
    if xyxy is None:
        return out
    class_ids = getattr(prediction, "class_id", None)
    scores = getattr(prediction, "confidence", None)
    for index in range(len(xyxy)):
        score = float(scores[index]) if scores is not None else 1.0
        if score < SCORE_THRESHOLD:
            continue
        out.append(
            {
                "box": [float(v) for v in xyxy[index]],
                "score": score,
                "class_id": int(class_ids[index]) if class_ids is not None else 0,
            }
        )
    return out


# ── the run ─────────────────────────────────────────────────────────────────


def interf0_handler():
    import cv2

    video_path = INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"
    if not video_path.exists():
        candidates = sorted(INPUT_PATH.rglob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"no video under {INPUT_PATH}")
        video_path = candidates[0]
    print(f"video: {video_path}", flush=True)

    models = load_models()
    capture = cv2.VideoCapture(str(video_path))

    boxes_out = []
    frame_index = 0
    batch, batch_indices = [], []

    def flush(batch, batch_indices):
        if not batch:
            return
        height, width = batch[0].shape[:2]
        per_view = []
        for model in models:
            raw = model.predict(batch, threshold=SCORE_THRESHOLD)
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            per_view.append([detections_of(p) for p in raw])
            if FLIP_TTA:
                mirrored = [image[:, ::-1].copy() for image in batch]
                raw_flip = model.predict(mirrored, threshold=SCORE_THRESHOLD)
                if not isinstance(raw_flip, (list, tuple)):
                    raw_flip = [raw_flip]
                per_view.append(
                    [unflip_horizontal(detections_of(p), width) for p in raw_flip]
                )
        for offset, index in enumerate(batch_indices):
            fused = filter_frame(
                weighted_box_fusion([view[offset] for view in per_view])
            )
            for detection in fused:
                x1, y1, x2, y2 = detection["box"]
                name = GC_NAME[CATEGORIES[detection["class_id"] % len(CATEGORIES)]]
                boxes_out.append(
                    {
                        # clockwise from top-left; the third value is 0.5 always,
                        # as the automated evaluator requires.
                        "name": f"slice_nr_{index}_{name}",
                        "corners": [
                            [x1, y1, 0.5],
                            [x2, y1, 0.5],
                            [x2, y2, 0.5],
                            [x1, y2, 0.5],
                        ],
                        "probability": detection["score"],
                    }
                )

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        batch_indices.append(frame_index)
        frame_index += 1
        if len(batch) >= BATCH_SIZE:
            flush(batch, batch_indices)
            batch, batch_indices = [], []
    flush(batch, batch_indices)
    capture.release()

    print(f"{frame_index} frames, {len(boxes_out)} boxes", flush=True)

    write_json_file(
        location=OUTPUT_PATH / "surgical-tools.json",
        content={
            "type": "Multiple 2D bounding boxes",
            "boxes": boxes_out,
            "version": {"major": 1, "minor": 0},
        },
    )
    return 0


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def run():
    return {("endoscopic-robotic-surgery-video",): interf0_handler}[get_interface_key()]()


if __name__ == "__main__":
    sys.exit(run())
