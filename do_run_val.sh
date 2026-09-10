#!/usr/bin/env bash
# ============================================================================
# do_run_val.sh — run the submission container on a cat1 validation video and
#                 render the boxes on top of it.
#
# WHAT THIS IS AND IS NOT
#   The cat1 videos are IN THE TRAINING SET of the shipped weights, so the boxes
#   this produces are NOT an accuracy measurement — the model has seen them. It
#   is a GEOMETRY check: are the boxes on the clevis, is the class plausible, is
#   the count per frame sane? For an honest number use the 7-fold protocol in
#   algorithms/run_cascade_kfold.sh (0.3654 +/- 0.0860).
#
#   It is also the reason this exists at all: the first Grand Challenge run
#   returned 99.3% "needle driver" on a 1280x720 test video, while the same
#   models on 640x512 cat1 footage classify correctly. Running here and looking
#   at the result is how that difference is diagnosed.
#
# USAGE
#   ./do_run_val.sh                 # video 3_fps1 (shortest, 297 frames)
#   ./do_run_val.sh 6_fps1          # the hard one: GT boxes 4.7% of frame area
#   VIDEO_DIR=/path ./do_run_val.sh 1_fps1
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The weights are not in git (127.7 MB each, above GitHub's per-file limit) and
# the forward pass runs with --network none, so they must be on disk first.
# Idempotent: a no-op once they are present.
echo "=+= Ensuring model weights are present"
"${SCRIPT_DIR}/fetch_weights.sh"

VIDEO_ID="${1:-5_fps1}"
VIDEO_DIR="${VIDEO_DIR:-$SCRIPT_DIR/../../data/cat1_gt}"
# Must match the tag do_build.sh actually builds. The previous default here
# was "Submission algoritm category 1 final phase" -- a tag that never
# existed (typo, spaces, capitals), so every run died with rc=125 unless the
# caller happened to export DOCKER_IMAGE_TAG themselves.
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-example-algorithm-category-1-final-phase}"
GPUS="${GPUS:-all}"
WORK="$SCRIPT_DIR/val_run/$VIDEO_ID"

[[ -f "$VIDEO_DIR/$VIDEO_ID.mp4" ]] || {
    echo "ERROR: no $VIDEO_DIR/$VIDEO_ID.mp4 — set VIDEO_DIR" >&2; exit 1; }

rm -rf "$WORK"; mkdir -p "$WORK/input" "$WORK/output"
cp "$VIDEO_DIR/$VIDEO_ID.mp4" "$WORK/input/endoscopic-robotic-surgery-video.mp4"
cat > "$WORK/input/inputs.json" <<'JSON'
[{"interface": {"slug": "endoscopic-robotic-surgery-video"}}]
JSON
# The container runs as an unprivileged user that is not the caller; without this
# the inference completes and then dies writing its result.
chmod 777 "$WORK/output"

echo "=+= inference on $VIDEO_ID"
start=$SECONDS
docker run --rm --platform=linux/amd64 --gpus "$GPUS" \
    --volume "$WORK/input":/input:ro \
    --volume "$WORK/output":/output \
    --volume "$SCRIPT_DIR/model":/opt/ml/model:ro \
    "$DOCKER_IMAGE_TAG"
echo "=+= $((SECONDS - start))s"

RESULT="$WORK/output/surgical-tools.json"
[[ -s "$RESULT" ]] || { echo "ERROR: no $RESULT" >&2; exit 1; }

echo "=+= what came out"
python3 - "$RESULT" <<'PY'
import collections, json, re, statistics, sys
boxes = json.load(open(sys.argv[1]))["boxes"]
frames = collections.Counter()
classes = collections.Counter()
for box in boxes:
    match = re.match(r"^slice_nr_(\d+)_(.+)$", box["name"])
    frames[int(match.group(1))] += 1
    classes[match.group(2)] += 1
counts = sorted(frames.values())
print(f"  {len(boxes)} boxes over {len(frames)} frames")
print(f"  per frame: mean {statistics.fmean(counts):.2f}  max {counts[-1]}"
      f"   (cat1 ground truth: 2.19, max 3)")
print(f"  classes: {dict(classes.most_common())}")
if len(classes) == 1:
    print("  WARNING: a single class — the class collapse seen on the challenge video")
PY

# The container writes the Grand Challenge shape; every renderer and scorer in
# this repo reads COCO. Convert before drawing anything.
COCO="$WORK/output/coco_${VIDEO_ID}.json"
python3 "$SCRIPT_DIR/algorithms/gc_to_coco.py" \
    --input "$RESULT" --output "$COCO" --video-id "$VIDEO_ID"
if [[ -f "$SCRIPT_DIR/algorithms/overlay_endovis_gt_boxes.py" ]]; then
    echo "=+= rendering overlay"
    # Rendered INSIDE the image: it already carries opencv, and requiring the
    # host to have it is what made this step fail the first time it ran.
    # Rendered INSIDE the image, and WITHOUT --user: the image installs its
    # wheels with `pip --user` into /home/user/.local, and /home/user is 0750, so
    # no other uid can even traverse into it — PYTHONPATH cannot rescue that.
    # Running as the image's own user is the only thing that sees opencv, so the
    # output directory has to be writable by it.
    REPO_ROOT=$( cd -- "$VIDEO_DIR/../.." &> /dev/null && pwd )
    mkdir -p "$WORK/video" && chmod 777 "$WORK/video"
    docker run --rm \
        --volume "$REPO_ROOT":"$REPO_ROOT" --volume "$SCRIPT_DIR":"$SCRIPT_DIR" \
        --workdir "$SCRIPT_DIR" --entrypoint python -e MPLCONFIGDIR=/tmp/mpl \
        "$DOCKER_IMAGE_TAG" algorithms/overlay_endovis_gt_boxes.py \
        --data-dir "$VIDEO_DIR" --annotations "$VIDEO_DIR" --video-id "$VIDEO_ID" \
        --pred-coco "$COCO" --pred-min-score 0.0 \
        --gt-color 0,255,0 --pred-color 255,0,0 \
        --output-dir "$WORK/video" --force \
      && echo "=+= video: $WORK/video" \
      || echo "  (overlay failed; the JSON in $WORK/output is still valid)"
fi
