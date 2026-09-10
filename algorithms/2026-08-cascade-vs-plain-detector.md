# Clevis detection for SurgVU26 cat1: what we measured, 2026-08-14/15

## The question

The pipeline's target is MICCAI SurgVU26 Category 1: per-frame **clevis** box plus
instrument class, ranked by COCO mAP@[0.5:0.05:0.95], class-aware. The incumbent
approach in this repo was a three-stage patch cascade — detect the whole
instrument, classify the crop, localise the clevis *inside* that crop — resting
on the premise that a clevis (median 73×67 px, 1.4% of a 640×512 frame) is too
small to detect directly and needs crop-and-zoom.

**That premise is wrong.** A plain detector trained directly on clevis boxes
beats the cascade by 1.7×, and the elaborations built to justify the cascade were
measured inert.

## Protocol (identical for every number below)

- 7-fold **leave-2-out** over the 7 public cat1 videos, from
  `clevis_pseudo_labeler.patch_localizer.folds.build_folds(val_size=2)`. Every
  video is validation exactly twice. This is the partition the incumbent 0.194
  was measured on.
- Ground truth: `data/cat1_gt`, 11,323 boxes over 5,178 frames, 640×512.
- Scorer: `scripts/aggregate_cat1_folds.py` (which wraps
  `benchmark_endovis_boxes`), `eval_scope: ground-truth`, `class_aware: true`.
- Frames materialised once with
  `clevis-yolo-prepare --input-format endovis --endovis-root data/cat1_gt` into
  `exports/cascade/cat1_clevis_yolo` — 5,178 images / 11,323 boxes, matching GT
  exactly.

**Reading the scorer's report**: it produces a 7-variant × 7-fold matrix. Only
the **diagonal** is honest — variant *f* is the only model that never saw fold
*f*'s validation videos. Off-diagonal cells reach 0.44 because they score models
on videos they trained on.

## Results

### Headline

| approach | mAP@[.5:.95] | source |
|---|---|---|
| YOLOv8n (sibling repo, same protocol) | 0.1738 | reference |
| **patch cascade (incumbent)** | **0.194 ± 0.079** | `runs/patch_localizer/cat1_kfold_eval` |
| RT-DETR-L (sibling repo) | 0.2203 | reference |
| plain 14-class RF-DETR-Medium | **0.3241 ± 0.0907** | `runs/cascade_kfold/scored_clevis14` |
| best-ever sibling repo (triple ensemble + TTA) | 0.3317 | reference |
| 3-model ensemble | 0.3476 ± 0.0817 | `scored_ensemble3_final` |
| 4-model ensemble | 0.3535 ± 0.0801 | `scored_ensemble4` |
| 4-model ensemble + flip TTA | 0.3607 ± 0.0837 | `scored_ensemble4_tta` |
| **6-model ensemble + flip TTA** | **0.3654 ± 0.0860** | `scored_ensemble6_tta` |

Mean recall rises monotonically at every ensemble step: 0.789 → 0.836 → 0.844 →
0.856 → 0.861, so the gains are recovered boxes rather than re-ranked scores.

**0.3654 is 1.88× the incumbent and +10.2% over the sibling repo's best-ever**,
which itself used a triple ensemble plus TTA.

Ensemble returns diminish but stay positive: +0.0235 (3 members), +0.0059 (4),
+0.0072 (flip TTA), +0.0047 (6). Variance stops improving after four members
(0.0801 → 0.0860), so past that point the gain is mean-only.

### Classification moved into the detector

Measured with `scripts/measure_patch_ceiling.py`, same scorer and settings:

| variant | crop classifier (DINOv3) | detector-native 14-class |
|---|---|---|
| oracle box + predicted class, macro per-class | 0.3487 | **0.8513** |
| oracle box + predicted class, summary | 0.5425 | 0.8673 |
| oracle box + oracle class | 0.9839 | 0.9790 |
| recall ceiling (padded crop) | 0.980 | 0.9778 |
| class accuracy on covered detections | 0.706 | 0.9282 |

Coverage is unchanged (0.980 vs 0.978), so this isolates classification. The
class ceiling moved from 0.349 to 0.851; geometry became the binding constraint
again.

### Things that did NOT work

| experiment | result | verdict |
|---|---|---|
| OCR roster correction of the training corpus | 53 of 23,266 boxes changed (0.23%) | already applied upstream by the auto-labeling loop |
| 28-class control (whole instrument + clevis as duplicated classes) | 0.3331 ± 0.0854 clevis-only | indistinguishable from 0.3381 base; auxiliary supervision is inert |
| EndoVis 2018 wrist pretrain (1,197 frames) | 0.3441 (+0.006) | noise |
| SAR-RARP50 wrist pretrain (12,400 frames) | 0.3400 (+0.002) | noise — 10× the data gave *less* |
| mixed EndoVis+SAR-RARP pretrain (prova A) | 0.3471 (+0.009) | best single model, still within noise |
| mixed + SurgVU as 2nd class (prova B) | 0.3372 ± 0.1081 | **below base**, highest variance |
| mixed + SurgVU in the same class (prova C) | 0.3451 ± 0.0763 | 2nd best single, **lowest variance** |

Single-model ranking, best first: **A mixed 0.3471 > C one-class 0.3451 >
EndoVis 0.3441 > SAR-RARP 0.3400 > base 0.3381 > B two-class 0.3372.**

A PREDICTION THAT WAS WRONG, recorded because it is informative: C was expected
to be the worst — it collapses boxes differing twentyfold in area into one class
— and it came second with the lowest variance, while B (which keeps them apart)
was the only variant *below* baseline. A plausible reading, offered as hypothesis
not conclusion: separating the two classes spends capacity on a distinction the
14-class finetune then discards, whereas collapsing them is simply more evidence
of "where an object is", which is what a pretrain should teach. The B-C gap is
0.008 against stdevs of ~0.08, so this needs its own experiment to settle.

**The external datasets pay as ensemble diversity, not as knowledge.** Alone they
are worth +0.002 to +0.009. But the ensemble's members *are* those pretrains, and
together they are worth +0.023.

Note the pretrains' own validation scores are not comparable to each other: a
wrist-only pretrain validates on wrist boxes (0.67), while a mix containing the
SurgVU whole-instrument corpus validates on a population that includes large,
easy boxes (0.80–0.84). Only the cat1 k-fold compares them.

## Method notes worth keeping

**Weighted Box Fusion, not NMS** (`clevis_pseudo_labeler/cascade/fusion.py`). NMS
keeps one box per cluster and discards the rest, throwing away exactly what an
ensemble is for — that several models independently found the same object. WBF
averages cluster coordinates weighted by score and scales by how many *distinct*
models contributed. Contributions are counted by model index, not by box: one
model firing twice is not a consensus of two.

**Horizontal flip only for TTA** (`cascade/tta.py`). A surgical scene has no
canonical left/right — instruments enter from either side — so mirroring is
label preserving. It *does* have a fixed up/down, so a vertical flip would ask
the model about a scene that never occurs. Each TTA view enters WBF as its own
ensemble member, so agreement between views earns the same bonus as agreement
between models.

**Each TTA/ensemble member must share the fold partition.** The three ensemble
members for fold *f* are the fold-*f* checkpoints of three different runs, all
trained on the same train videos. Mixing folds would leak.

## Traps that cost real time

1. **`--pretrain-weights` silently shrinks the class head.** RF-DETR adopts the
   *checkpoint's* class count and `_align_num_classes_from_dataset` then keeps it
   over the dataset's. A 1-class wrist checkpoint therefore trains a 1-class head
   against labels with ids up to 13 and dies in a device-side assert — visible
   only as a C++ stack trace at interpreter shutdown, preceded by two easily
   missed warnings. Fixed by adding `--num-classes`, which
   `jobs/run_cascade_kfold.sh` now **derives from the dataset** rather than
   exposing as a forgettable flag.
2. **The runtime image is older than the checkout.** Its `config.py` rejects
   `clevis_eval`, and its installed `rfdetr` has no `check_val_every_n_epoch` on
   `TrainConfig` (which is `extra="forbid"`) — so `configs/default.yaml` and
   `configs/hpc_pair.yaml` would both fail against it. Every job must bind
   `clevis_pseudo_labeler/`, `configs/` and `scripts/` over `/app`.
3. **`docker service create` has no `--entrypoint`.** CPU-only stages run as a
   plain `docker run`.
4. **Per-fold `--dataset-dir` is mandatory.** `prepare_rfdetr_dataset` defaults
   its adapter directory beside the dataset.yaml, and all seven fold yamls share
   a parent — without it, seven jobs materialise into the same directory
   concurrently.
5. **Fold dataset.yaml paths must be absolute.** RF-DETR resolves a relative
   split entry against the yaml's own directory, and the fold yaml lives under
   the run's output root, not the dataset root.
6. **Prediction files must be named `coco_*.json`**, not `*_coco.json` — the
   directory glob in `_expand_prediction_paths` looks for that prefix, while the
   ground truth uses the opposite convention.
7. **`video_id` belongs inside each COCO image record.** `_image_key` prefers it
   over anything derived from the filename.
8. **EndoVis 2018 wrist is class 3; SAR-RARP50 wrist is class 2.** Verified
   against the pixels, not assumed.
9. **EndoVis 2018 annotates the wrist in only 4 of 18 sequence-splits**, and
   there in ~100% of frames. The other 14 have zero wrist pixels despite showing
   instruments — a different labelling protocol, not clevis-free scenes.
   Exporting them as negatives would teach the model that instruments have no
   clevis.

## Where the remaining loss is

Fold 5 (validation = videos 6 and 7) scores 0.19 against a 0.36 mean and drags
everything down. Video 6 was already known to be a geometric outlier: median GT
box 4.70% of frame area versus ~1.3% elsewhere. Understanding that failure is
probably worth more than another point of average.

## Artifacts

- Datasets: `exports/cascade/{cat1_clevis_yolo, cat1_dual28, auto_labeling-087-ocr,
  endovis2018_wrist_split, sar_rarp50_wrist, surgvu_as_class0, surgvu_as_class1}`
- Runs: `runs/cascade_kfold/*`, `runs/cascade_pretrain/*`, `runs/rfdetr_cascade/*`
- Scored reports: `runs/cascade_kfold/scored_*/kfold_report.json`
- Code: `clevis_pseudo_labeler/cascade/`, `jobs/run_cascade_{experiment,kfold}.sh`,
  `scripts/{predict_folds_coco,predict_folds_ensemble,convert_endovis2018_wrist_to_yolo,
  convert_sar_rarp50_wrist_to_yolo,download_sar_rarp50,build_dual_class_dataset,
  correct_labels_from_ocr_roster,remap_yolo_class}.py`

## Recommendation

The two-decoder cascade in `clevis_pseudo_labeler/cascade/{model,criterion}.py`
(implemented, 80 tests green, training module not written) would now have to beat
**0.3607**, not 0.194 — and both hypotheses that motivated it have been weakened
by measurement: auxiliary whole-instrument supervision is inert, and the
crop-and-zoom premise is contradicted. Decide whether to finish it before
investing further.

Cheaper directions with evidence behind them: more ensemble members (gains were
still positive at four), multi-scale TTA on top of the flip, and the fold-5 /
video-6 failure mode.
