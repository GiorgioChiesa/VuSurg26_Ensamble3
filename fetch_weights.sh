#!/usr/bin/env bash
#
# Fetch the three ensemble checkpoints from Hugging Face into model/ensemble/.
#
# WHY THIS RUNS ON THE HOST, NOT IN THE CONTAINER
#
# The forward pass runs with `--network none` — Grand Challenge gives the
# algorithm no internet, and do_test_run.sh/do_run_val.sh reproduce that
# faithfully. So the weights must be on disk before any container starts; a
# download from inside inference.py would work on your laptop and fail on the
# platform, which is the worst kind of difference.
#
# Idempotent: a file already present with the right checksum is left alone, so
# this is safe to call at the top of every run script.
#
#   ./fetch_weights.sh            # download what is missing
#   ./fetch_weights.sh --check    # verify only, download nothing
#   ./fetch_weights.sh --force    # re-download everything
set -euo pipefail

REPO=${CLEVIS_HF_REPO:-SGio16/Ensamble3}
BASE="https://huggingface.co/${REPO}/resolve/main"
DEST="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/model/ensemble"

# sha256 of each checkpoint. A truncated or corrupted download is otherwise
# indistinguishable from a good one until the model produces nonsense: torch
# raises deep inside unpickling, far from the cause.
FILES=(
    "clevis_sarrarp.pth 41d0c9a77a67f960d07515f8a88ba2adcb4a454b1bb10ee02b478808b0b93bdc"
    "clevis_mixA.pth    539d70280c4c6462515a0d79e3a0bb2d666f8a83336ae3858dc5ea04bf1db13a"
    "clevis_mixC.pth    33a42a8372fa0ee31e8e8ebc6cc2fd190b3ec9cfb40279b9e78ee1f2e1395637"
)

CHECK_ONLY=""; FORCE=""
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "fetch_weights: unknown option $arg" >&2; exit 2 ;;
    esac
done

command -v curl >/dev/null || { echo "fetch_weights: curl is required" >&2; exit 2; }
mkdir -p "$DEST"

ok=0; fetched=0; failed=0
for entry in "${FILES[@]}"; do
    read -r name want <<<"$entry"
    path="$DEST/$name"

    if [[ -f "$path" && -z "$FORCE" ]]; then
        got=$(sha256sum "$path" | cut -d' ' -f1)
        if [[ "$got" == "$want" ]]; then
            echo "  ok       $name"
            ok=$((ok + 1)); continue
        fi
        # A corrupt file and an absent one need different words: the first
        # means someone's disk or download went wrong, the second means nothing
        # has been fetched yet. Reporting both as "missing" sends you looking in
        # the wrong place.
        corrupt=1
        echo "  MISMATCH $name (on disk $got)" >&2
    fi

    if [[ -n "$CHECK_ONLY" ]]; then
        [[ -n "${corrupt:-}" ]] || echo "  MISSING  $name" >&2
        unset corrupt
        failed=$((failed + 1)); continue
    fi
    unset corrupt

    echo "  fetching $name (127.7 MB) from $REPO"
    # Download beside the target, not onto it: an interrupted transfer must not
    # leave a half file where a later run would find it and trust its presence.
    # -L: the resolve endpoint redirects to a CDN.
    if ! curl -fL --retry 3 --retry-delay 2 --progress-bar \
              -o "$path.part" "$BASE/$name"; then
        echo "  FAILED   $name (download)" >&2
        rm -f "$path.part"; failed=$((failed + 1)); continue
    fi

    got=$(sha256sum "$path.part" | cut -d' ' -f1)
    if [[ "$got" != "$want" ]]; then
        echo "  FAILED   $name (checksum: expected $want, got $got)" >&2
        rm -f "$path.part"; failed=$((failed + 1)); continue
    fi
    mv "$path.part" "$path"
    echo "  ok       $name"
    fetched=$((fetched + 1))
done

echo "fetch_weights: $ok present, $fetched downloaded, $failed failed"
[[ "$failed" -eq 0 ]] || exit 1
