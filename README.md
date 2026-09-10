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

**The `.pth` files are not in this repository**: each is 127.7 MB, above
GitHub's 100 MB per-file limit.

<!-- TODO: replace with the Hugging Face repo URL once it exists -->
Weights are hosted separately — see the release notes. Place the three
checkpoints under `model/ensemble/` before building:

```
model/ensemble/clevis_sarrarp.pth
model/ensemble/clevis_mixA.pth
model/ensemble/clevis_mixC.pth
```

`inference.py` globs **only** `model/ensemble/`, deliberately: the full model
package also carries whole-instrument detectors, a different object entirely,
and a recursive glob would enrol them as ensemble members and flood every frame
with instrument-sized boxes labelled as clevises.

## Running it

```bash
./do_build.sh                 # build the image
./do_test_run.sh              # run on the sample under test/ (needs a real video)
./do_run_val.sh               # run over a directory of validation videos
./do_save.sh                  # export the image as a tarball for upload
```

The base image is `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`. A CPU base is
not viable: the ensemble runs six forward passes per frame and would time out.
`torch >= 2.5` is required for a non-obvious reason — `transformers` 5.x silently
disables its PyTorch integration below that version, and the backbone mixins
`rfdetr` imports are then never exported, so the container dies on
`cannot import name 'BackboneConfigMixin'`, which looks like a `transformers`
version problem and is not.

### Post-processing configuration

Three environment variables, applied in this order: score floor, cross-class
suppression, per-frame cap. The shipped values disable the first two:

| Variable | Shipped | Library default |
|---|---|---|
| `CLEVIS_MIN_SCORE` | `0` | `0.3` |
| `CLEVIS_CROSS_CLASS_IOU` | `1.0` (off) | `0.55` |
| `CLEVIS_MAX_PER_FRAME` | `10` | `3` |

The sweep's optimum for the cap is `100`, not `10`; shipping `10` cost about
0.0023 mAP locally. It survived review because on the challenge's own videos the
cap truncates only 1.4 % of frames — on a held-out video it truncates 28.4 %,
because a less certain model emits a longer tail. It is the one setting we would
change.

## Layout

```
Dockerfile          container definition, including the post-processing env vars
inference.py        the whole inference path: fusion, TTA, filtering, output
requirements.txt    pinned runtime dependencies
do_*.sh             build / test / validate / export
model/README.md     model card: what shipped, how it was trained, provenance
algorithms/         training, evaluation and dataset-preparation code
val_run/            prediction outputs on the seven released videos (evidence)
test/               sample input/output pair for a smoke run
```

`val_run/` holds the JSON predictions only. The rendered overlay videos and the
challenge input footage are deliberately **not** redistributed here: the source
videos are governed by the SurgVU data agreement and are available from the
challenge organisers.

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
