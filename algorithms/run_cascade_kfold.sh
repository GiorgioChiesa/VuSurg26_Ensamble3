#!/usr/bin/env bash
# ============================================================================
# run_cascade_kfold.sh — train one detector per fold, all folds at once
#
# THE PROTOCOL IS NOT NEGOTIABLE
#   The partition comes from clevis_pseudo_labeler.patch_localizer.folds.build_folds
#   with val_size=2 over the 7 cat1 videos: 7 folds, every video is validation
#   exactly twice. That is the same partition the incumbent 0.194 +/- 0.079 was
#   measured on (runs/patch_localizer/cat1_kfold_eval/kfold_report.json). Change
#   it and the comparison this whole branch exists for stops being a comparison.
#
# WHY THE FOLDS COST NO DISK
#   The source dataset keeps every video in its own directory
#   (images/<split>/<video_id>/...), and RF-DETR's adapter accepts a LIST of
#   image directories per split. A fold is therefore just a dataset.yaml naming
#   five directories -- not a seventh copy of 5,178 images.
#
# THE RACE THIS AVOIDS
#   prepare_rfdetr_dataset defaults its adapter directory to
#   <dataset.yaml>.parent/rfdetr_dataset. Every fold yaml lives in the same
#   parent, so without an explicit per-fold --dataset-dir all seven jobs would
#   materialise into the SAME directory concurrently and silently train on each
#   other's symlinks.
#
# USAGE
#   ./jobs/run_cascade_kfold.sh --dataset exports/cascade/cat1_dual28 \
#                               --output-root runs/cascade_kfold/dual28
#   ./jobs/run_cascade_kfold.sh --dataset ... --folds 0,3 --dry-run
# ============================================================================
set -uo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

CONFIG=${CONFIG:-configs/cat1_cascade.yaml}
IMAGE=${CLEVIS_HPC_IMAGE:-eidos-service.di.unito.it/nunziante/clevis-pseudo-labeler:latest}
RF_HOME_HOST=${RF_HOME_HOST:-$repo_dir/runs/.roboflow}
# 20, not 40. Stage 1 peaked at epoch 14 and then declined monotonically
# (0.7887 -> 0.7658 @19 -> 0.7410 @24); the rest was compute spent making the
# weights worse.
EPOCHS=${EPOCHS:-20}
MODEL=${MODEL:-rfdetr-medium}
DATASET=; OUTPUT_ROOT=; FOLDS=; DRY_RUN=${DRY_RUN:-}
# Optional geometry warm start (e.g. the EndoVis 2018 wrist pretrain).
#
# MEASURED, not assumed: a pretrain checkpoint does NOT get its class head grown
# to the dataset's class count. load_pretrain_weights adopts the CHECKPOINT's
# count and _align_num_classes_from_dataset then keeps it over the dataset's, so
# a 1-class wrist checkpoint trains a 1-class head against labels carrying ids
# up to 13 and dies in a device-side assert -- visible only as a C++ stack trace
# at interpreter shutdown, preceded by two easily-missed warnings. That is what
# killed all seven folds of the first EndoVis run.
#
# --num-classes is therefore DERIVED from the dataset below and passed whenever
# a pretrain is used. It is not a knob; forgetting it is the bug.
PRETRAIN_WEIGHTS=${PRETRAIN_WEIGHTS:-}
# Optional. Unset means "whatever the config says" (42), i.e. exactly the
# behaviour of every sweep run before this flag existed. Set it to build an
# ensemble whose diversity comes from seeds rather than from different pretrains.
SEED=${SEED:-}
# Optional. Swarm keeps a task's logs in the daemon and `docker service rm` takes
# them with it, so a completed sweep leaves nothing behind unless they are
# streamed out while it runs.
LOG_DIR=${LOG_DIR:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --folds)       FOLDS="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --pretrain-weights) PRETRAIN_WEIGHTS="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --log-dir)     LOG_DIR="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$DATASET" && -n "$OUTPUT_ROOT" ]] || {
    echo "usage: $0 --dataset DIR --output-root DIR [--folds 0,1] [--epochs N]" >&2
    echo "          [--model M] [--pretrain-weights P] [--seed N] [--log-dir DIR]" >&2
    exit 2; }
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"

mkdir -p "$RF_HOME_HOST" "$OUTPUT_ROOT"

# ── write one dataset.yaml per fold ─────────────────────────────────────────
mapfile -t FOLD_IDS < <(
docker run --rm --user "$(id -u):$(id -g)" -v "$repo_dir:$repo_dir" \
    -v "$repo_dir/clevis_pseudo_labeler:/app/clevis_pseudo_labeler:ro" \
    -w "$repo_dir" --entrypoint python -e MPLCONFIGDIR=/tmp/mpl "$IMAGE" -c "
import pathlib, yaml
from clevis_pseudo_labeler.cascade.fold_yamls import fold_dataset_yaml
from clevis_pseudo_labeler.patch_localizer.folds import build_folds
root = pathlib.Path('$DATASET').resolve(); out = pathlib.Path('$OUTPUT_ROOT')
names = yaml.safe_load((root/'dataset.yaml').read_text())['names']
splits = {d.name: s for s in ('train','val') for d in (root/'images'/s).iterdir() if d.is_dir()}
for f in build_folds(sorted(splits), val_size=2):
    doc = fold_dataset_yaml(train_videos=f['train_videos'], val_videos=f['val_videos'],
                            video_splits=splits, names=names, root=str(root))
    d = out / f\"fold{f['fold']}\"; d.mkdir(parents=True, exist_ok=True)
    (d/'dataset.yaml').write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f['fold'])
") || { echo "ERROR: could not write the fold partition" >&2; exit 1; }

[[ ${#FOLD_IDS[@]} -gt 0 ]] || { echo "ERROR: no folds produced" >&2; exit 1; }
if [[ -n "$FOLDS" ]]; then IFS=',' read -ra FOLD_IDS <<< "$FOLDS"; fi
NUM_CLASSES=$(awk '/^nc:/ {print $2; exit}' "$DATASET/dataset.yaml")
[[ -n "$NUM_CLASSES" ]] || { echo "ERROR: no nc: in $DATASET/dataset.yaml" >&2; exit 1; }
echo "folds: ${FOLD_IDS[*]}  (dataset $DATASET, $EPOCHS epochs, $MODEL, nc=$NUM_CLASSES)"

# ── submit every fold at once ───────────────────────────────────────────────
declare -A SERVICE_OF
declare -a TAIL_PIDS=()
for fold in "${FOLD_IDS[@]}"; do
    fold_dir="$OUTPUT_ROOT/fold$fold"
    args=(rfdetr-train --config "$CONFIG"
          --data "$fold_dir/dataset.yaml"
          --dataset-dir "$fold_dir/adapter"
          --output-dir "$fold_dir/run"
          --model "$MODEL" --epochs "$EPOCHS")
    if [[ -n "$PRETRAIN_WEIGHTS" ]]; then
        args+=(--pretrain-weights "$PRETRAIN_WEIGHTS" --num-classes "$NUM_CLASSES")
    fi
    if [[ -n "$SEED" ]]; then
        args+=(--seed "$SEED")
    fi
    if [[ -n "$DRY_RUN" ]]; then
        printf '  [dry-run] fold %s: clevis-pseudo-labeler' "$fold"; printf ' %q' "${args[@]}"; printf '\n'
        continue
    fi
    name="chiesa-kfold-$$-f$fold"
    docker service create --detach --quiet --with-registry-auth \
        --name "$name" --restart-condition none \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,source=$repo_dir,destination=$repo_dir" \
        --mount "type=bind,source=$RF_HOME_HOST,destination=/rfhome" \
        --mount "type=bind,source=$repo_dir/clevis_pseudo_labeler,destination=/app/clevis_pseudo_labeler,readonly" \
        --mount "type=bind,source=$repo_dir/configs,destination=/app/configs,readonly" \
        --mount "type=bind,source=$repo_dir/scripts,destination=/app/scripts,readonly" \
        --workdir "$repo_dir" \
        --env PYTHONUNBUFFERED=1 --env RF_HOME=/rfhome --env MPLCONFIGDIR=/tmp/mpl \
        --env HF_HUB_OFFLINE=1 \
        --generic-resource gpu=1 \
        "$IMAGE" "${args[@]}" >/dev/null \
        && { SERVICE_OF[$fold]=$name; echo "  fold $fold -> $name"; } \
        || echo "  fold $fold: SUBMIT FAILED" >&2
    if [[ -n "$LOG_DIR" && -n "${SERVICE_OF[$fold]:-}" ]]; then
        docker service logs -f "$name" >"$LOG_DIR/fold$fold.log" 2>&1 &
        TAIL_PIDS+=($!)
    fi
done
[[ -n "$DRY_RUN" ]] && exit 0

cleanup() {
    for p in "${TAIL_PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done
    for n in "${SERVICE_OF[@]}"; do docker service rm "$n" >/dev/null 2>&1; done
}
trap cleanup EXIT

# ── poll them all together ──────────────────────────────────────────────────
# An EXPLICIT counter, not ${#DONE[@]}: under `set -u` bash treats the length of
# an empty associative array as an unbound variable, so the loop aborted on its
# first evaluation and the EXIT trap tore down all seven jobs one second after
# submitting them.
declare -A DONE=()
submitted=${#SERVICE_OF[@]}
finished=0
failed=0
while (( finished < submitted )); do
    for fold in "${!SERVICE_OF[@]}"; do
        [[ -n "${DONE[$fold]:-}" ]] && continue
        state=$(docker service ps "${SERVICE_OF[$fold]}" --no-trunc --format '{{.CurrentState}}' 2>/dev/null | head -1)
        case "$state" in
            Complete*)
                DONE[$fold]=ok; finished=$((finished + 1))
                echo "  [ok]     fold $fold  ($state)" ;;
            Failed*|Rejected*|Orphaned*)
                DONE[$fold]=fail; finished=$((finished + 1)); failed=1
                echo "  [FAILED] fold $fold  ($state)" >&2
                docker service logs "${SERVICE_OF[$fold]}" 2>&1 | tail -25 >&2 ;;
        esac
    done
    (( finished < submitted )) && sleep 60
done

echo "=== esito ==="
for fold in $(printf '%s\n' "${!DONE[@]}" | sort -n); do echo "  fold $fold: ${DONE[$fold]}"; done
exit $failed
