# Supplementary Material

**Frozen High-Resolution Inference for Cross-City Object Detection: An AI City Challenge 2026 Study**

This archive contains the retained implementation and documented settings for
reconstructing the interventions (L0–L4) reported in the paper. One run-level
value could not be recovered and is disclosed explicitly (the L3 warmup, §3).
Model weights and the challenge data are **not** included: the weights are large
(~130 MB each) and the Track 6 datasets and hidden benchmark are only accessible
inside the air-gapped Hafnia Training-as-a-Service (TaaS) platform and may not be
redistributed.

Training and benchmark inference run as a Hafnia trainer package (`python scripts/train.py …`,
`python scripts/benchmark.py …`). `pyproject.toml` + `uv.lock` pin every package
version; `Dockerfile` and `.python-version` define the runtime.

---

> **Self-contained package.** This ZIP includes `RUN_MANIFEST.md` (per-run
> provenance: verbatim commands, settings and scores), `RESULTS.md` (all reported
> metrics) and `NOTICE.md` (upstream provenance and modifications). No external
> link is required to audit any reported number.

## 1. Contents

```
SUPPLEMENTARY.md                         this file
code/
  README.md, LICENSE                     trainer-package readme (required by Dockerfile/pyproject)
  Dockerfile, .python-version
  pyproject.toml, uv.lock                exact package versions
  configs/                               per-configuration settings (L0–L4 + base)
    base_704.yaml
    l0_base_704.yaml
    l1_frozen_1120.yaml
    l2_rect_1280x736.yaml
    l3_finetune_1120.yaml
    l4_grayworld_1120.yaml
  scripts/
    train.py                 training + warm-start fine-tuning entry point
    benchmark.py             hidden-benchmark inference / prediction export
    wbf_ensemble.py          optional standalone WBF utility; not used for reported results
    package_predictions.py   submission packaging (confidence floor + top-100 truncation + coord rounding)
    create_pretrained_model.py
    visualize.py             local visualisation (never touches the hidden benchmark)
    train.schema.json, benchmark.schema.json   CLI argument schemas
  src/trainer_object_detection/
    aug_presets.py           dg_crosscity, dg_crosscity_v2 augmentation definitions
    wbf.py                   optional WBF engine; not used for reported results
    wrapped_model.py         inference wrapper: resize, hflip TTA, gray-world, num_select
    utils.py
  extra_experiments/         NOT used to produce the reported results
    export_onnx.py, stress_eval.py, sweep_resolution.py
```

---

## 2. Base training configuration (Table 1) — `configs/base_704.yaml`

RF-DETR-Large (rfdetr 1.8.1; DINOv2 ViT-S windowed backbone), trained on the
Track 6 source-city split.

| Item | Value |
|---|---|
| Model | RF-DETR-Large (`rfdetr` 1.8.1) |
| Backbone | DINOv2 ViT-S, windowed |
| Train resolution | 704 × 704 |
| Epochs | 80 |
| Optimizer | AdamW |
| Decoder learning rate | 7 × 10⁻⁵ |
| Encoder learning rate | 1 × 10⁻⁴ (`--lr-encoder`) |
| Scheduler | cosine (`lr_scheduler="cosine"`) |
| Warmup | 1.0 epoch (`--warmup-epochs 1.0`) |
| `lr_min_factor` | 0.05 |
| Effective batch | 16 (batch 1 × grad-accum 16) |
| Precision | mixed (AMP) |
| `num_select` | 300 |
| Augmentation | `dg_crosscity_v2` (see `aug_presets.py`) |
| Augmentation backend | cpu (Albumentations) |
| Weight decay | 1 × 10⁻⁴ (rfdetr `TrainConfig` default; not overridden) |
| EMA decay | 0.993 (`ema_decay`; `ema_tau` = 100) |
| Random seed | none set (`seed = None`; runs are not seeded — see paper Limitations, n = 1) |
| Checkpoint selection | best source-validation AP; higher of EMA / regular (`checkpoint_best_total`) |

`scripts/train.py` now exposes `--warmup-epochs`, `--lr-encoder`, and
`--lr-min-factor` as CLI arguments (default warmup 1.0) so base and fine-tune
runs are reproducible from the command line. Weight decay (1 × 10⁻⁴), EMA decay
(0.993), and the absence of a fixed seed are the rfdetr `TrainConfig` defaults
for the pinned version (`uv.lock`).

---

## 3. L3 — 1120px warm-start fine-tuning — `configs/l3_finetune_1120.yaml`

Warm-start from the shared **704-trained checkpoint** (the same checkpoint used for the
inference-only configurations L0–L2 and L4).

| Item | Value |
|---|---|
| Warm-start checkpoint | shared 704-trained checkpoint (`checkpoint_best_total` of the base run, i.e. the higher-scoring of EMA/regular; exported as `run1_best.zip`) |
| Additional epochs | 6 |
| Train resolution | 1120 × 1120 |
| Optimizer | AdamW (re-initialised) |
| Decoder learning rate | 1.5 × 10⁻⁵ (`--learning-rate`) |
| Encoder learning rate | 1 × 10⁻⁴ (`--lr-encoder`) |
| Scheduler | cosine (re-initialised) |
| Effective batch | 16 (batch 1 × grad-accum 16) |
| Precision | mixed (AMP) |
| Augmentation | `dg_crosscity_v2` |
| Weight decay | 1 × 10⁻⁴ |
| EMA decay | 0.993 |
| Random seed | none set |
| Evaluated checkpoint | higher-scoring of EMA / regular by source-validation AP (`checkpoint_best_total`), evaluated at 1120 × 1120 |

**Disclosed gap — L3 warmup.** The exact `warmup_epochs` used for this 6-epoch
fine-tune was **not separately retained**. The base 80-epoch run used a 1.0-epoch
warmup, which is the pipeline default; the
`0.25` that appears in earlier snapshots of `train.py` belonged to a *later*
2-epoch self-training experiment, not to L3. We therefore do not assert a single
L3 warmup value in the paper. To reproduce with an explicit value:

```
python scripts/train.py \
  --model-path ./pretrained_models/run1_best.zip \
  --epochs 6 --batch-size 1 --grad-accumulation-steps 16 \
  --learning-rate 1.5e-5 --lr-encoder 1e-4 --warmup-epochs 1.0 \
  --resolution 1120 --aug-preset dg_crosscity_v2 \
  --inference-model-name checkpoint_best_total
```

---

## 4. Inference configurations (L0–L4) — `configs/l*.yaml`

The YAML files under `configs/` are **documentation snapshots** of the retained
per-configuration settings; they are not directly consumed by `train.py` /
`benchmark.py` (which take CLI arguments). Each section below and §7 gives the
corresponding CLI command.

All inference-only configurations use one shared 704-trained checkpoint; only the
listed factor changes. Confidence threshold, top-k (`num_select`), resize /
interpolation, coordinate restoration, and box-clipping live in
`src/trainer_object_detection/wrapped_model.py` and are exposed as CLI flags.

| Config | Change | Key settings |
|---|---|---|
| **L0** | base 704 inference | resolution 704 |
| **L1** | frozen 1120 inference (training-free) | resolution 1120 (square, direct resize; PE interpolated), no parameter update |
| **L2** | rectangular inference | resolution `1280x736`, parsed HxW → 1280(H) × 736(W) |
| **L3** | 1120px fine-tuning | see §3 |
| **L4** | gray-world white balancing | per-image gray-world channel-mean normalisation (`--inference.grayworld`) |

Common inference settings across all reported runs: an effective `num_select`
of 300, predictions rescaled back to the original image size, and boxes clipped
to bounds. Two settings were **not** common: L0 used the compiled
native-resolution path while L1–L4 ran eagerly, and L0/L1/L2/L4 used confidence
threshold `0.01` while L3 used `0.05` (see the run-record audit in §4).
Evaluation separately uses COCO `maxDets = 100` per image (an evaluator setting,
not a model inference option). L0 uses square 704 × 704 inference at the training grid (no
positional-embedding interpolation); L1 and L4 use square 1120 × 1120 inference
with positional-embedding interpolation; L2 uses a rectangular direct resize.

**Resolution string order.** Square resolutions are given as a single integer.
Rectangular resolutions are parsed by the benchmark CLI in `H×W` order
(`InferenceConfig.shape_hw()` in `src/trainer_object_detection/wrapped_model.py`
returns `(parts[0], parts[1])`, forwarded to `RFDETR.predict(shape=(h, w))`).
The recorded L2 configuration string is `1280x736`, so the actual model input
shape was **1280(H) × 736(W)** — a tall rectangular input. L2 is therefore a
rectangular-orientation test and is **not** an aspect-ratio-preserving or
native-aspect input: orientation, aspect ratio, image shape, and pixel count all
differ from the square 1120 setting at once.

**Verbatim platform commands.** The following are the exact command strings
recorded by the Hafnia platform for each reported run, recovered from the
experiment records after the challenge closed. `run1_best.zip` is the shared
704-trained checkpoint (`checkpoint_best_total`, i.e. the higher-scoring of the
EMA and regular weights by source-validation AP) exported from the base training
run. Flags not shown were left at their CLI defaults.

| ID | Platform experiment | Date | Command |
|---|---|---|---|
| L0 | `run1 benchmark crosscity Scale` | 2026-07-03 | `python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip` |
| L1 | `run1 probe res1120sq` | 2026-07-05 | `python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip --inference.resolution 1120 --inference.no-compile` |
| L2 | `run1 probe res1280x736` | 2026-07-04 | `python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip --inference.resolution 1280x736 --inference.no-compile` |
| L3 | `ft1120b v5 6ep autobench` | 2026-07-10 | `python scripts/train.py --model-path ./pretrained_models/run1_best.zip --epochs 6 --batch-size 1 --grad-accumulation-steps 16 --learning-rate 1.5e-5 --resolution 1120 --aug-preset dg_crosscity_v2 --inference-model-name checkpoint_best_total --inference-config.resolution 1120 --inference-config.no-compile --inference-config.threshold 0.05` |
| L4 | `run1 gw1120 probe` | 2026-07-09 | `python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip --inference.resolution 1120 --inference.grayworld --inference.no-compile` |

Base training run (produces `run1_best.zip`), `Large run1 80ep`, 2026-06-28:

```
python scripts/train.py --model-path ./pretrained_models/RFDETRLarge.zip \
    --epochs 80 --batch-size 1 --grad-accumulation-steps 16 \
    --learning-rate 7e-5 --aug-preset dg_crosscity_v2 \
    --inference-model-name checkpoint_best_total
```

**Three points where the run records differ from a naive reading of the above.**

1. **Confidence threshold was *not* uniform.** L0, L1, L2 and L4 used the CLI
   default `threshold = 0.01`. The L3 run's built-in evaluation pass was launched
   with an explicit `--inference-config.threshold 0.05`. Because COCO AP is
   rank-based under a per-image detection cap, the higher threshold discards
   low-confidence detections that could still have contributed to the
   precision–recall tail, so the difference is expected to work *against* L3. The
   magnitude is unmeasured: the evaluation server is closed and L3 cannot be
   re-scored at `0.01`. The paper reports the L1–L3 comparison as confounded by
   this in addition to the optimization differences.
2. **`num_select` was never passed explicitly.** The CLI default is `None`, which
   falls back to the model's built-in `300`, so the effective value is `300` in
   every reported run — but no reported command pins it. (A separate,
   *unreported* experiment, `run1 ns600 hflipTTA`, did use
   `--inference.num-select 600 --tta-hflip`; it is not part of L0–L4.)
3. **Compiled inference.** L0 ran with the compiled path enabled (the CLI
   default), because it evaluates at the checkpoint's native square resolution.
   L1, L2 and L4 pass `--inference.no-compile`, which a resolution override
   requires: `optimize_for_inference()` pins the compiled graph to the
   training-time square resolution, so any override must run eagerly. This is a
   consequence of the override, not an independently chosen setting.

Encoder learning rate (`1e-4`) and the cosine minimum-LR factor were left at the
`train.py` defaults in every run; `--learning-rate` sets the decoder LR only. See
`benchmark.schema.json` and `train.schema.json` for all flags.

**Confidence threshold (two distinct stages).**
1. *Model inference threshold* = `0.01` (`benchmark.schema.json` default) — a
   deliberately low threshold that retains low-confidence detections for COCO AP
   evaluation.
2. *Submission-packaging floor* = `0.015`, applied by
   `scripts/package_predictions.py` (default `--confidence-floor 0.015`,
   overridable) together with per-class top-100 truncation (`--per-class-top-k 100`),
   5-decimal coordinate rounding, and 4-decimal confidence rounding.

| Stage | Value | Role |
|---|---:|---|
| Model inference threshold | 0.01 | filters the detections the model emits before serialization |
| Submission-packaging floor | 0.015 | additional filter applied during submission packaging |
| Per-class truncation | 100 | max predictions kept per class |
| Coordinate precision | 5 decimals | serialized coordinate rounding |
| Confidence precision | 4 decimals | serialized confidence rounding |

These two thresholds are distinct: `0.01` governs which detections the model
emits; `0.015` is an additional filter applied only during submission packaging.
All of these packaging operations are treated as part of the evaluated submission
pipeline; their individual effects on the COCO AP were **not** measured
separately. Per-class top-100 truncation is a packaging size-control step and is
distinct from the evaluator's COCO `maxDets = 100` per-image cap. Run
`package_predictions.py --help` from `code/scripts/`. COCO scoring: AP over IoU
0.50:0.05:0.95, `maxDets = 100` per image.

**Retained WBF utility (not in the reported configuration set).** A
multi-resolution WBF configuration was explored during the challenge but is **not**
part of the paper's evaluated set. The WBF code is retained only as optional
tooling: `src/trainer_object_detection/wbf.py` is the in-process fusion engine for
`benchmark.py`'s single-pipeline ensemble path (`--ensemble-extra` / `--tta-hflip`,
which fuses multiple model archives or hflip views in one run), and
`scripts/wbf_ensemble.py` is a standalone offline fusion tool (its parquet writer
is also reused by `package_predictions.py`). Neither is used to produce the
reported L0–L4 results.

---

## 5. Augmentation: `dg_crosscity_v2`

Defined in `src/trainer_object_detection/aug_presets.py` as
`AUG_DG_CROSSCITY_V2` (ISP-style photometric randomisation + soft geometry;
Albumentations 2.0.8, pinned in `uv.lock`). Applied at both base training and L3
fine-tuning.

---

## 6. Package versions

`code/pyproject.toml` (declared deps) and `code/uv.lock` (fully resolved,
hash-pinned) fix every version, including `rfdetr==1.8.1`, `hafnia`,
`albumentations==2.0.8`, and PyTorch. `Dockerfile` + `.python-version` define the
runtime image. Build check:

```
cd code
uv sync --locked
python scripts/train.py --help
python scripts/benchmark.py --help
```

---

## 7. Reproduction run order

1. **Base training** — `python scripts/train.py --model-path <rf-detr-large-2026.zip> --epochs 80 --resolution 704 --learning-rate 7e-5 --lr-encoder 1e-4 --warmup-epochs 1.0 --aug-preset dg_crosscity_v2 --batch-size 1 --grad-accumulation-steps 16` (inside Hafnia, `eccv-cross-city` dataset). Produces the shared 704-trained checkpoint (`checkpoint_best_total`, exported as `run1_best.zip`). The actual platform run passed neither `--resolution` (704 is the model default) nor `--lr-encoder`/`--warmup-epochs` (defaults); see the verbatim command in §4.
2. **L3 fine-tuning** — §3 launch command, warm-starting from the shared 704-trained checkpoint.
3. **Inference (L0–L2, L4)** — attach the shared checkpoint and run `python scripts/benchmark.py` with the per-config flags (§4 / `configs/`).
4. **Download** the experiment `/model` output and extract the predictions; package with `package_predictions.py`; **submit** the bundle to the AI City Challenge evaluation server.

Because the server reports only a single aggregate COCO AP over the hidden
source/target mixture, each configuration is measured by one server submission;
there is no local target ground truth (see paper, Limitations).

---

## 8. Notes

- **No weights / no data.** Model checkpoints and the Track 6 datasets are not
  redistributable and are omitted; the code reproduces the pipeline given access
  to the Hafnia platform and the allowed pretrained RF-DETR weights.
- **`extra_experiments/`** (`export_onnx.py`, `stress_eval.py`,
  `sweep_resolution.py`) were part of the working repository
  but are **not** used to produce the reported L0–L4 results. `sweep_resolution.py`
  is the resolution-sweep scaffold listed as future work (paper §7); it was never
  run (compute budget exhausted after the challenge).
- **Pseudo-label / self-training path.** `train.py` retains an optional
  `--pseudo-label` self-training path (teacher pass over the unlabeled TEST
  split). It was exercised only as a separate exploratory experiment and was
  **not** used to produce any of the reported L0–L4 results. No claim is made
  here about its permissibility under the Track 6 rules.
- **`visualize.py`** never reads the hidden benchmark images (platform rule).
- **Disclosed reconstruction gap:** the exact L3 warmup value (§3). Everything
  else is either an explicit setting here or a pinned dependency default.
