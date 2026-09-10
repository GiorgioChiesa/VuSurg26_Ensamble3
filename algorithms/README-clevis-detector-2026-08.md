# Clevis detector for SurgVU26 cat1 — full experiment record

**Dates**: 2026-08-14 → 2026-08-16 · **Branch**: `trio` · **Host**: hssh1 (Swarm, 3×8 A40)

Results and analysis: [`2026-08-cascade-vs-plain-detector.md`](2026-08-cascade-vs-plain-detector.md).
This file is the *record*: what was asked, what was done in order, and where
every artifact lives.

---

## 1. The original request

Pre-train a DETR segmenter on the whole training set to recognise the **whole
instrument**, then k-fold finetune on the validation set to narrow the boxes to
the **clevis**. Evaluate mixed training with EndoVis 2017 and SAR-RARP50. Also
evaluate a single model doing instrument detection + classification + clevis
detection (duplicated classes, or two decoders in cascade/parallel), keeping
classification accurate — optionally using the OCR heuristic, which knows which
instrument is installed even when it is not visible.

Training corpus specified: Label Studio project **"Auto labeling"**, policy
**reviewed-or-confident**, confidence threshold **0.87**.

## 2. What the measurements did to that plan

Three of the plan's premises were tested and did not survive:

1. **"The clevis is too small to detect directly."** A plain detector trained on
   clevis boxes scores 0.324 against the crop-and-zoom cascade's 0.194.
2. **"Whole-instrument supervision helps the clevis."** The 28-class control
   (both objects as duplicated classes) scored 0.3331 against a 0.3381 baseline
   — inert.
3. **"The OCR roster is an untapped lever."** It is already applied upstream by
   `jobs/classifier_auto_pair.sh`; re-applying it moved 0.23% of boxes.

Two premises held, one in an unexpected form:

- **Classification had to move into the detector.** Class ceiling 0.349 → 0.851.
- **External datasets pay — as ensemble diversity, not as knowledge.** Alone
  +0.002 to +0.009; as ensemble members, +0.023.

Final: **0.3654 ± 0.0860** (6-model ensemble + flip TTA), 1.88× the incumbent.

EndoVis **2017** is not on this machine (`data/endovis2017` does not exist);
EndoVis **2018** is, and carries the same wrist annotation, so it was used
instead. SAR-RARP50 was downloaded.

## 3. Chronological protocol

Every step below is reproducible from the paths in §4–§5.

### Phase 0 — training corpus
1. Export the Label Studio project by exact title:
   `python jobs/export_review.py --title "Auto labeling" --out <bundle>/reviewed_cascade.json`
   → 8,016 tasks, 7,703 annotated.
2. **Deduplicate**: 3 tasks carried multiple completed annotations (identical
   content, UI double-submits). `sync_resumable_annotations` raises on these, so
   the export was de-duplicated locally. *The shared Label Studio project was
   not modified* — the same error will hit the auto-labeling loop's stage 7.
3. Apply the gate:
   `clevis-labelstudio-apply --incomplete-policy reviewed-or-confident --min-detection-confidence 0.87`
   → 11,062 frames / 23,266 boxes (7,703 reviewed + 3,359 admitted by the gate,
   535 dropped).
   **Only the detection floor is passed**: `_meets_confidence_thresholds` treats
   a *missing* field as failure, so adding the other two floors at 0.0 would
   admit zero unreviewed frames.
4. OCR roster correction:
   `python scripts/correct_labels_from_ocr_roster.py` → 20 relabels, 33 drops
   (0.23%). Kept because it is free; it is not a lever.

### Phase 1 — classification into the detector
5. Train 14-class RF-DETR-Medium on the corpus:
   `./jobs/run_cascade_experiment.sh --only 1`. Peaks at **epoch 14**, then
   overfits monotonically — stop at 15–20, not 40.
6. Run it over cat1 and measure the gate:
   `./jobs/run_cascade_experiment.sh --only 2` … `--only 4`
   → `oracle_box_pred_class` 0.3487 → **0.8513**.

### Phase 2 — the direct detector and its k-folds
7. Materialise cat1 as YOLO (5,178 frames / 11,323 boxes, exact GT match):
   `clevis-yolo-prepare --input-format endovis --endovis-root data/cat1_gt --class-mode source`
8. 7-fold leave-2-out, all folds in parallel:
   `./jobs/run_cascade_kfold.sh --dataset exports/cascade/cat1_clevis_yolo --output-root runs/cascade_kfold/clevis14`
9. Score through the common scorer:
   `python scripts/predict_folds_coco.py --kfold-root <run> --dataset exports/cascade/cat1_clevis_yolo`
   then `python scripts/aggregate_cat1_folds.py --config configs/cat1_cascade.yaml --variant fold0=<run>/fold0/preds …`
   **Read the diagonal** — the report is a 7-variant × 7-fold matrix and only
   variant *f* on fold *f* never saw its validation videos.

### Phase 3 — controls and external data
10. 28-class control: `scripts/build_dual_class_dataset.py` → k-fold → 0.3331.
11. EndoVis 2018: `scripts/convert_endovis2018_wrist_to_yolo.py` (wrist = class 3)
    → pretrain → k-fold → 0.3441.
12. SAR-RARP50: `scripts/download_sar_rarp50.py` (figshare, open, CC BY-NC-SA 4.0)
    → `scripts/convert_sar_rarp50_wrist_to_yolo.py` (wrist = **class 2**, not 3)
    → pretrain → k-fold → 0.3400.
13. Three mixes (A: wrist only; B: SurgVU as 2nd class; C: SurgVU collapsed into
    the same class), via `scripts/remap_yolo_class.py` and the `mixed_*.yaml`
    files → 0.3471 / 0.3372 / 0.3451.

### Phase 4 — ensembles
14. `python scripts/predict_folds_ensemble.py --kfold-root <run> … [--flip-tta]`
    fuses several runs' fold-*f* checkpoints with Weighted Box Fusion.
    3 members 0.3476 → 4 members 0.3535 → +flip TTA 0.3607 → 6 members + TTA
    **0.3654**.
15. Annotated videos:
    `python scripts/overlay_endovis_gt_boxes.py --pred-coco <fold>/preds --pred-min-score 0.3`

---

## 4. Where the datasets are

All under `/mnt/fast-scratch/chiesa/clevis/exports/cascade/`.

| dataset | images | boxes | nc | what it is |
|---|---|---|---|---|
| `cat1_clevis_yolo` | 5,178 | 11,323 | 14 | **the target**: cat1 GT clevis boxes, cat1 taxonomy, one dir per video |
| `auto_labeling-087` | 11,062 | 23,266 | 14 | SurgVU whole-instrument corpus at the 0.87 gate |
| `auto_labeling-087-ocr` | 11,062 | 23,233 | 14 | the same, OCR-roster corrected |
| `cat1_dual28` | 5,178 | 23,828 | 28 | the duplicated-class control |
| `endovis2018_wrist_split` | 1,197 | 2,882 | 1 | EndoVis 2018 wrist, own train/val |
| `sar_rarp50_wrist` | 12,400 | 23,513 | 1 | SAR-RARP50 wrist |
| `surgvu_as_class0` / `surgvu_as_class1` | 11,062 | 23,233 | 1 | SurgVU corpus remapped for mixes C / B |
| `mixed_wrist.yaml` | — | — | 1 | mix A: EndoVis + SAR-RARP (lists dirs, copies nothing) |
| `mixed_wrist_2class.yaml` | — | — | 2 | mix B: + SurgVU as class 1 |
| `mixed_wrist_1class.yaml` | — | — | 1 | mix C: + SurgVU collapsed to class 0 |

Images in the derived datasets are **symlinks**, so the sources must stay:

| source | path | note |
|---|---|---|
| cat1 GT | `data/cat1_gt/` | 7 videos + `*_coco.json`, 168 MB |
| SurgVU videos | `/mnt/fast-scratch/shared/datasets/medical/visual/surgvu24_videos/` | 326 GB, read-only |
| OCR roster labels | `data/surgvu_ocr_labels/` | 155 cases |
| EndoVis 2018 | `/mnt/fast-scratch/shared/datasets/medical/visual/endovis2018/` | 3.0 GB, read-only |
| SAR-RARP50 | `data/sar_rarp50/` | 51 GB (zips + unpacked), CC BY-NC-SA 4.0 |
| LS review workspace | `exports/labelstudio/auto_labeling_cascade/` | private copy; `media/` symlinks to the shared workspace |

## 5. Where the model weights are

`checkpoint_best_ema.pth` is the selected checkpoint everywhere (128 MB each).
`last.ckpt` and `checkpoint_<n>.ckpt` are Lightning states (~510 MB), not needed
for inference.

**Per-fold detectors** — 7 checkpoints each, at
`runs/cascade_kfold/<run>/fold<0..6>/run/checkpoint_best_ema.pth`:

| run | init | k-fold mAP |
|---|---|---|
| `clevis14` | COCO | 0.3381 |
| `clevis14_ev18` | EndoVis 2018 | 0.3441 |
| `clevis14_sarrarp` | SAR-RARP50 | 0.3400 |
| `clevis14_mixed` | mix A | **0.3471** |
| `clevis14_mix2c` | mix B | 0.3372 |
| `clevis14_mix1c` | mix C | 0.3451 |
| `dual28` | COCO, 28 classes | 0.3331 (clevis-only) |

**The six `clevis14*` runs are the ensemble members.** Fold *f* of each is a
model that never saw fold *f*'s validation videos, which is what makes the
ensemble honest.

**Pre-trained geometry checkpoints** at `runs/cascade_pretrain/<name>/run/checkpoint_best_ema.pth`:
`endovis2018`, `sar_rarp50`, `mixed_wrist` (A), `mixed_2class` (B), `mixed_1class` (C).

**Phase-1 14-class detector** (the one that lifted the class ceiling):
`runs/rfdetr_cascade/stage1_14class/checkpoint_best_ema.pth`.

**RF-DETR COCO pretrain cache**: `runs/.roboflow/` — keep it, or every job
re-downloads 386 MB.

Disk: `runs/cascade_kfold` 93 GB, `runs/cascade_pretrain` 7.5 GB,
`runs/rfdetr_cascade` 1.9 GB. Most of it is Lightning `last.ckpt` files that can
be deleted without losing any result.

## 6. Results, reports and videos

- Scored k-fold reports: `runs/cascade_kfold/scored_*/kfold_report.json`
  (`scored_clevis14`, `scored_ensemble3_final`, `scored_ensemble4`,
  `scored_ensemble4_tta`, **`scored_ensemble6_tta`** = the best).
- Class-ceiling reports: `runs/clevis_patch/00_ceiling` (crop classifier) and
  `runs/clevis_patch/10_ceiling_detector_class` (detector-native).
- **Annotated videos**: `runs/cascade_kfold/ensemble6_tta/videos/fold<N>/<video>_gt_vs_pred.mp4`
  — 14 videos, GT in green, predictions in blue, both with class names, filtered
  at score ≥ 0.3 (predictions are generated at 0.05, which is 9.6 boxes/frame and
  unreadable). Each cat1 video appears in two folds, produced by different models.

## 7. Code added

`clevis_pseudo_labeler/cascade/`: `classification_view`, `coco_predictions`,
`fold_yamls`, `fusion` (WBF), `tta`, `endovis2018`, `sar_rarp50`, `remap`,
`dual_class`, plus `model`/`criterion`/`config` for the two-decoder cascade
(implemented, **training module never written**).

`scripts/`: `predict_folds_coco`, `predict_folds_ensemble`,
`correct_labels_from_ocr_roster`, `detections_to_classification`,
`build_dual_class_dataset`, `convert_endovis2018_wrist_to_yolo`,
`download_sar_rarp50`, `convert_sar_rarp50_wrist_to_yolo`, `remap_yolo_class`.

`jobs/`: `run_cascade_experiment.sh` (4 stages), `run_cascade_kfold.sh`
(7 folds in parallel).

`configs/cat1_cascade.yaml` — deliberately separate from `hpc_pair.yaml`, which
is rewritten by the auto-labeling loop.

Tests: 148 passing. One pre-existing failure,
`test_rfdetr_annotate_roster_reads_config_defaults`, verified unrelated by
stashing the change and seeing it fail identically.

`clevis_pseudo_labeler/rfdetr/train.py` and `cli.py` gained `--num-classes`;
see trap 1 in the results document.

## 8. What was NOT done

- The **cascade's training module**. Model and losses exist with 80 green tests;
  nothing wires them to the Lightning trainer, so the two-decoder cascade was
  never trained. It would now have to beat 0.3654.
- **EndoVis 2017** — not on this machine.
- **graphify update** — the manifest does not match (all 4,253 files report as
  changed even with `kind="ast"`), so it would need a full rebuild: 824 documents
  and 107 papers through subagents, as no `GEMINI_API_KEY` is set.
- A **final submission model**. The k-fold mean is the honest estimate; a model
  to submit must be retrained on all 7 videos, or the 7 fold-ensembles kept.
  After training on cat1 there is no honest held-out set left.
