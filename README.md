# VuSurg26 — Ensamble 3

Inference container submitted to **MICCAI SurgVU 2026, Category 1** (clevis
detection and instrument classification in robot-assisted surgical video).

This entry placed **6th among the competing teams** on the final-phase test set,
scoring **0.5385 mAP**; the leading entry scored 0.5720.

## What it does

Category 1 asks for the *clevis* — the pin-and-jaw joint nearest the distal tip
of a robotic instrument — to be localised in each frame and assigned one of
fourteen instrument types. The clevis covers a median 1.41 % of the frame and
looks much the same across instruments, so detection is easy and classification
is the hard part.

The container runs a **three-member ensemble of RF-DETR-Medium detectors**, each
one fine-tuned from a different external pre-train, fused with weighted box
fusion and horizontal-flip test-time augmentation — six forward passes per frame.

| Checkpoint | Stage-1 external pre-train | Classes |
|---|---|---|
| `clevis_sarrarp.pth` | SAR-RARP50 wrist | 1 |
| `clevis_mixA.pth` | EndoVis 2018 + SAR-RARP50 | 1 |
| `clevis_mixC.pth` | wrist + whole instrument | 2 |

Three members, not six: all 63 non-empty subsets of six trained models were
scored offline on a seven-fold leave-two-out protocol. The best three-member
subset scores 0.6392 (0.6414 with flip TTA) against 0.6413 for all six — a
difference well inside the protocol's standard error of ±0.022, at half the
inference cost.

## Measured performance

Seven-fold leave-two-out on the released challenge videos, mAP@[.5:.05:.95],
class-aware:

| | value |
|---|---|
| mAP@[.5:.95] | **0.6414** |
| AP@0.5 | 0.9441 |
| Precision @ IoU 0.5 | 0.322 |
| Recall @ IoU 0.5 | 0.968 |
| Median matched IoU | 0.854 |

Recall is high and precision deliberately low: the container emits every
surviving detection. Average precision integrates the precision–recall curve, so
a low-scoring false positive ranked below the true positives costs almost
nothing while a discarded true positive costs a whole object. A 96-point sweep
of the post-fusion knobs confirms it — the detector library's default
configuration (`0.3 / 3 / 0.55`) costs **0.0396 mAP** against emitting
everything.

> Note on provenance: the 0.6414 figure describes models cross-validated on a
> 9,814-frame pseudo-label corpus, while the shipped weights were trained two
> days later on 10,486 frames. It is the best available estimate of these
> weights, not a measurement of them, and the discrepancy runs in the favourable
> direction. See `model/README.md`.

## Model weights

The three checkpoints live on Hugging Face, not in this repository: each is
127.7 MB, above GitHub's 100 MB per-file limit.

**https://huggingface.co/SGio16/Ensamble3**

You do not need to fetch them by hand. `fetch_weights.sh` downloads them into
`model/ensemble/`, verifies each against its SHA-256, and does nothing if they
are already there:

```bash
./fetch_weights.sh            # download whatever is missing
./fetch_weights.sh --check    # verify only, download nothing
./fetch_weights.sh --force    # re-download everything
```

`do_test_run.sh` and `do_run_val.sh` call it themselves, so in practice you never
run it directly. The repository is public and no Hugging Face token is needed.

The download happens **on the host, before any container starts**, and that is
deliberate: the forward pass runs with `--network none`, because Grand Challenge
gives the algorithm no internet. Downloading from inside `inference.py` would
work on your machine and fail on the platform.

`inference.py` globs **only** `model/ensemble/`. That is not laziness: the full
model package also carries whole-instrument detectors, a different object
entirely, and a recursive glob would enrol them as ensemble members and flood
every frame with instrument-sized boxes labelled as clevises.

## Running it

### Prerequisites

- Docker with the NVIDIA container runtime — the ensemble runs six forward
  passes per frame and does not finish on CPU
- One NVIDIA GPU, roughly 6 GB free
- `curl` and `sha256sum` on the host, for the weight download
- About 1.5 GB of disk for the image, plus 383 MB for the weights

### 1. Build

```bash
./do_build.sh
```

Builds `example-algorithm-category-1-final-phase` from
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`.

`torch >= 2.5` is required for a reason two steps removed from its symptom:
`transformers` 5.x silently disables its PyTorch integration on older torch, and
with it gone the backbone mixins `rfdetr` imports are never exported. The
container then dies on `cannot import name 'BackboneConfigMixin'`, which looks
like a `transformers` version problem and is not.

### 2. Smoke test

```bash
./do_test_run.sh
```

Fetches the weights if needed, rebuilds, and runs one forward pass over
`test/input/interf0/`, writing `test/output/interf0/surgical-tools.json`.

**The sample video is not in this repository** — challenge footage is not
redistributed here (see *Data and what is not here*). Drop any endoscopic `.mp4` at
`test/input/interf0/endoscopic-robotic-surgery-video.mp4` before running.

On a shared machine, pin a GPU instead of taking all of them:

```bash
GPUS='"device=1"' ./do_test_run.sh
```

### 3. Run on a video and look at the boxes

```bash
VIDEO_DIR=/path/to/videos ./do_run_val.sh <video_id>
```

This runs the container over `<video_id>` and renders the predictions on top of
the footage, writing under `val_run/<video_id>/`:

```
input/inputs.json          the Grand Challenge interface descriptor
output/surgical-tools.json the raw submission-format predictions
output/coco_<id>.json      the same boxes in COCO form, for scoring
video/<id>_gt_vs_pred.mp4  ground truth against predictions, side by side
```

**What this is and is not.** The seven challenge videos are *in the training
set* of these weights, so the boxes are not an accuracy measurement — the model
has seen them. It is a **geometry check**: are the boxes on the clevis, is the
class plausible, is the count per frame sane? For an honest number use the
seven-fold protocol in `algorithms/run_cascade_kfold.sh`.

It earns its place: the first Grand Challenge run returned 99.3 % "needle
driver" on a 1280x720 test video while the same models classified correctly on
640x512 footage. Looking at the rendered video is how that was diagnosed.

### 4. Export for upload

```bash
./do_save.sh
```

Writes `<tag>_<build-timestamp>.tar.gz` for upload to Grand Challenge, then
re-reads what it wrote. That verification is not decoration: an earlier version
lacked `set -o pipefail`, and in `docker save | gzip > file` bash inspects only
gzip's exit status — a truncated 1.26 GB image was shipped while the script
printed success.

### Configuring post-processing

Three environment variables, applied in this order — score floor, cross-class
suppression, per-frame cap — set in the `Dockerfile` so changing them needs no
rebuild of the model:

| Variable | Shipped | Library default | Effect of the default |
|---|---|---|---|
| `CLEVIS_MIN_SCORE` | `0` | `0.3` | drops low-scoring boxes |
| `CLEVIS_CROSS_CLASS_IOU` | `1.0` (off) | `0.55` | suppresses overlapping boxes of different classes |
| `CLEVIS_MAX_PER_FRAME` | `10` | `3` | caps detections per frame |

Together the library defaults cost **0.0396 mAP** against emitting everything,
measured over a 96-point sweep. The sweep's optimum for the cap is `100`, not
`10`; shipping `10` cost about 0.0023 locally. It survived review because on the
challenge's own videos the cap truncates just 1.4 % of frames — on a held-out
video it truncates 28.4 %, because a less certain model emits a longer tail. It
is the one setting we would change.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `rc=125` from `do_run_val.sh` | image not built, or `DOCKER_IMAGE_TAG` overridden to a tag that does not exist |
| `cannot import name 'BackboneConfigMixin'` | torch below 2.5; `transformers` disabled its PyTorch integration |
| `no model checkpoint under /opt/ml/model` | weights absent — run `./fetch_weights.sh` |
| Every frame full of huge boxes | whole-instrument checkpoints ended up in `model/ensemble/` |
| Container times out | running on CPU; six forward passes per frame need a GPU |
| `fetch_weights` reports MISMATCH | a previous download was interrupted; re-run, it re-fetches |

## Data and what is not here

`val_run/` holds the **JSON predictions only**. The rendered overlay videos and
the challenge input footage are deliberately **not** redistributed here: the
source videos are governed by the SurgVU data agreement and are available from
the challenge organisers, who are the right people to get them from.

That is why `do_test_run.sh` needs you to supply a video, and why
`do_run_val.sh` takes a `VIDEO_DIR`. The predictions are published so that the
container's output can be inspected and re-scored without the footage; the
footage itself is not ours to hand out.

## Layout

```
Dockerfile          container definition, including the post-processing env vars
inference.py        the whole inference path: fusion, TTA, filtering, output
requirements.txt    pinned runtime dependencies
do_*.sh             build / test / validate / export
fetch_weights.sh    downloads the checkpoints from Hugging Face, checksum-verified
model/README.md     model card: what shipped, how it was trained, provenance
algorithms/         training, evaluation and dataset-preparation code
val_run/            prediction outputs on the seven released videos (evidence)
test/               sample input/output pair for a smoke run
```

## Citation

The dataset and challenge:

- Zia, A., Berniker, M., Nespolo, R., Zhang, X., Perreault, C., Wang, Z.,
  Mueller, B., Schmidt, R., Bhattacharyya, K., Liu, X., & Jarc, A. (2025).
  *Surgical Visual Understanding (SurgVU) Dataset*. arXiv:2501.09209
- Zia, A., et al. (2023). *Intuitive Surgical SurgToolLoc and SurgVU Challenges
  Results: 2022–2025*. arXiv:2305.07152

The detector:

- Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2026).
  *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*.
  ICLR. arXiv:2511.09554

The fusion rule:

- Solovyev, R., Wang, W., & Gabruseva, T. (2021). *Weighted boxes fusion:
  Ensembling boxes from different object detection models*. Image and Vision
  Computing, 107, 104117.
