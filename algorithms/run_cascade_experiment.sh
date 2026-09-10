#!/usr/bin/env bash
# ============================================================================
# run_cascade_experiment.sh — the CASCADE branch, stage by stage
#
# WHAT THIS BRANCH IS
#   Measured on the cat1 GT (runs/clevis_patch/00_ceiling/report.json): an oracle
#   clevis box with an ORACLE class scores 0.984 mAP, the same box with the REAL
#   crop classifier scores 0.349. Classification is the binding constraint. This
#   branch moves classification into the detector itself, then (stage 2) adds a
#   second decoder that narrows the box to the clevis.
#
#   Incumbent to beat: 0.194 +/- 0.079 mAP, 7-fold leave-2-out
#   (runs/patch_localizer/cat1_kfold_eval/kfold_report.json).
#
# USAGE
#   ./jobs/run_cascade_experiment.sh --list
#   ./jobs/run_cascade_experiment.sh --dry-run
#   ./jobs/run_cascade_experiment.sh --only 1
#
# MECHANICS THAT MATTER
#   * `docker service create --restart-condition none` NEVER converges, so an
#     attached client blocks forever even after the task succeeds. Submit with
#     --detach and poll. Same lesson as jobs/valuate_complete_pipeline.sh:160.
#   * The service name must be unique per submission: a completed service is not
#     auto-removed, and a fixed name collides with its own corpse.
#   * PYTHONUNBUFFERED=1 or the task looks frozen for 15+ minutes.
#   * Bind sources must exist on every node. /mnt/fast-scratch is GlusterFS and
#     resolves identically on hssh1/2/3; anything under /tmp does not.
#   * --user keeps the outputs owned by the caller. Without it the container
#     writes as root and the host user cannot even stat through its own runs/.
# ============================================================================
set -uo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

CONFIG=${CONFIG:-configs/cat1_cascade.yaml}
IMAGE=${CLEVIS_HPC_IMAGE:-eidos-service.di.unito.it/nunziante/clevis-pseudo-labeler:latest}
# Persist RF-DETR's COCO pretrain (medium is 386 MB) instead of re-downloading it
# on every submission.
RF_HOME_HOST=${RF_HOME_HOST:-$repo_dir/runs/.roboflow}
DRY_RUN=${DRY_RUN:-}
ONLY=

# EMA, not last: the run plateaus around epoch 14 and the EMA weights are what
# checkpoint_best_* selects on. metrics.csv is the record of that choice.
WEIGHTS=${WEIGHTS:-runs/rfdetr_cascade/stage1_14class/checkpoint_best_ema.pth}
CAT1_ANNOTATIONS=${CAT1_ANNOTATIONS:-runs/rfdetr_cascade/cat1_annotations}
# Resolved from data/cache_cat1_gt/0_sampling/active.json. That cache is owned by
# root and read-only to us, which is fine: stage 2 only reads the roster from it.
CAT1_ROSTER=${CAT1_ROSTER:-$(
    python3 - <<'PY' 2>/dev/null
import json, pathlib
root = pathlib.Path("data/cache_cat1_gt/0_sampling")
active = json.loads((root / "active.json").read_text())
print(root / "rosters" / f"{active['roster_id']}.json")
PY
)}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only)    ONLY="$2"; shift 2 ;;
        --list)
            echo "1  train the 14-class whole-instrument detector"
            echo "2  run it over the cat1 GT footage (conf 0.05, stride 1)"
            echo "3  present its class head as a classification source"
            echo "4  THE GATE: ceiling vs the crop classifier's 0.349"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$CAT1_ROSTER" ]] || { echo "ERROR: could not resolve the cat1 roster" >&2; exit 1; }

mkdir -p "$RF_HOME_HOST"

job_name=
cleanup_job() { [[ -n "$job_name" ]] && docker service rm "$job_name" >/dev/null 2>&1; }
trap cleanup_job EXIT

job() {
    if [[ -n "$DRY_RUN" ]]; then
        printf '  [dry-run] clevis-pseudo-labeler'; printf ' %q' "$@"; printf '\n'
        return 0
    fi

    job_name="chiesa-cascade-$$-$RANDOM"
    docker service create --detach --quiet --with-registry-auth \
        --name "$job_name" --restart-condition none \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,source=$repo_dir,destination=$repo_dir" \
        --mount "type=bind,source=$RF_HOME_HOST,destination=/rfhome" \
        `# The image ships its own INSTALLED copy of the package at /app, and it
         # is older than this checkout -- old enough that its config.py rejects
         # clevis_eval as an unknown key. Without these three overlays the job
         # silently runs the image's code and ignores every edit made here.` \
        --mount "type=bind,source=$repo_dir/clevis_pseudo_labeler,destination=/app/clevis_pseudo_labeler,readonly" \
        --mount "type=bind,source=$repo_dir/configs,destination=/app/configs,readonly" \
        --mount "type=bind,source=$repo_dir/scripts,destination=/app/scripts,readonly" \
        --workdir "$repo_dir" \
        --env PYTHONUNBUFFERED=1 \
        --env RF_HOME=/rfhome \
        `# --user means HOME is unwritable, and matplotlib then rebuilds its font
         # cache on every single run and says so at WARNING level.` \
        --env MPLCONFIGDIR=/tmp/mpl \
        --env HF_HUB_OFFLINE=1 \
        --generic-resource gpu=1 \
        "$IMAGE" "$@" >/dev/null || { echo "ERROR: submit failed" >&2; return 1; }

    echo "  submitted $job_name"
    local state=
    while true; do
        state=$(docker service ps "$job_name" --no-trunc --format '{{.CurrentState}}' 2>/dev/null | head -1)
        case "$state" in
            Complete*) echo "  $state"; break ;;
            Failed*|Rejected*|Orphaned*)
                echo "  $state" >&2
                docker service logs "$job_name" 2>&1 | tail -60 >&2
                cleanup_job; job_name=; return 1 ;;
            *) sleep 20 ;;
        esac
    done
    docker service logs "$job_name" 2>&1 | tail -25
    cleanup_job; job_name=
}

job_python() {
    # `docker service create` has no --entrypoint, and these stages need no GPU,
    # so they run as a plain local container instead of a Swarm task.
    if [[ -n "$DRY_RUN" ]]; then
        printf '  [dry-run] python'; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "$repo_dir:$repo_dir" \
        -v "$repo_dir/clevis_pseudo_labeler:/app/clevis_pseudo_labeler:ro" \
        -w "$repo_dir" --entrypoint python \
        -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/mpl \
        "$IMAGE" "$@"
}

stage_1() { # train the whole-instrument detector with 14 instrument-type classes
    # No flags beyond --config on purpose: every value is in the config file, so
    # the run is reproducible from that file alone rather than from shell history.
    job rfdetr-train --config "$CONFIG"
}

stage_2() { # run the 14-class detector over the cat1 GT footage
    # conf 0.05 and frame-stride 1 are NOT tunables here: they are what the patch
    # branch used (configs/cat1_patch.yaml), and the 0.980 containment ceiling and
    # the 0.349 class ceiling were both measured under them. Change either and the
    # comparison this whole stage exists for stops being a comparison.
    #
    # The output does NOT go to data/cache_cat1_gt/10_rfdetr_annotations: that
    # artifact belongs to the patch branch, is owned by root, and overwriting it
    # would silently invalidate every number already measured against it.
    job rfdetr-annotate-roster \
        --config "$CONFIG" \
        --weights "$WEIGHTS" \
        --roster "$CAT1_ROSTER" \
        --data-dir data/cat1_gt \
        --output-dir "$CAT1_ANNOTATIONS" \
        --mode track --tracker bytetrack \
        --conf 0.05 --frame-stride 1 \
        --inference-batch-size 32 --progress
}

stage_3() { # present the detector's class head as a classification source
    job_python scripts/detections_to_classification.py \
        --detections "$CAT1_ANNOTATIONS/track_annotations.json" \
        --output "$CAT1_ANNOTATIONS/detector_classification.json"
}

stage_4() { # THE GATE: does the detector's own class beat the crop classifier?
    # The number to beat is oracle_box_pred_class = 0.349 macro per-class
    # (runs/clevis_patch/00_ceiling/report.json). Read recall_ceiling_padded_crop
    # in the same report first: if it moved away from 0.980, coverage changed too
    # and the class comparison is no longer apples to apples.
    job_python scripts/measure_patch_ceiling.py \
        --config "$CONFIG" \
        --detections "$CAT1_ANNOTATIONS/track_annotations.json" \
        --classification "$CAT1_ANNOTATIONS/detector_classification.json" \
        --output-root runs/clevis_patch --run-id 10_ceiling_detector_class
}

stages=(1 2 3 4)
[[ -n "$ONLY" ]] && stages=("$ONLY")
for n in "${stages[@]}"; do
    echo "=== stage $n ==="
    start=$SECONDS
    if ! "stage_$n"; then
        printf '[FAILED]  stage %s (%dm %ds)\n' "$n" $(( (SECONDS-start)/60 )) $(( (SECONDS-start)%60 ))
        exit 1
    fi
    printf '[SUCCESS] stage %s (%dm %ds)\n' "$n" $(( (SECONDS-start)/60 )) $(( (SECONDS-start)%60 ))
done
