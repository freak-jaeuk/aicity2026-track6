# Run manifest — reported configurations L0–L4

Every score below is tied to the verbatim platform command in the same row.
These records provide **provenance**, not a newly executed reproduction: the
managed Hafnia benchmark and its evaluation server are closed, so none of these
runs can be re-executed or re-scored.

## Summary

| ID | Input | Checkpoint | Param. update | Threshold | Execution | AP |
|---|---|---|---|---:|---|---:|
| L0 | 704 × 704 | shared 704 ckpt | No | 0.01 | **compiled** | 0.3272 |
| **L1** | **1120 × 1120** | shared 704 ckpt | No | 0.01 | eager | **0.3654** |
| L2 | 1280(H) × 736(W) | shared 704 ckpt | No | 0.01 | eager | 0.3057 |
| L3 | 1120 × 1120, fine-tuned | warm-start from shared 704 ckpt | **Yes** | **0.05** | eager | 0.3470 |
| L4 | 1120 × 1120 + gray-world | shared 704 ckpt | No | 0.01 | eager | 0.3647 |

"Shared 704 ckpt" = `checkpoint_best_total` of the base training run (the
higher-scoring of the EMA and regular weights by source-validation AP), exported
as `run1_best.zip`. `num_select` was never passed explicitly in any reported run;
the effective value is the model default of 300 throughout.

**Two settings were not uniform.** L0 ran on the compiled native-resolution path
(the CLI default) while L1–L4 ran eagerly, which a resolution override forces.
L3's built-in evaluation pass used `--inference-config.threshold 0.05` while
every other run used the CLI default `0.01`. The threshold difference is expected
to work against L3, but its magnitude is unmeasured. The L1–L3 comparison is
therefore **not controlled**.

## Per-run records

### L0 — base 704 inference
- Platform experiment: `run1 benchmark crosscity Scale`, 2026-07-03, COMPLETED
- Config file: `code/configs/l0_base_704.yaml`
- Command:
  ```
  python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip
  ```
- Input 704 × 704 (model default; no `--inference.resolution` passed) · threshold 0.01 (default) · num_select 300 (default) · compiled · no parameter update
- AP 0.3272 · AP50 0.4391 · AP75 0.3335 · AP_S 0.0519 · AP_M 0.1528 · AP_L 0.4687
- AR@1 0.3568 · AR@10 0.5782 · AR@100 0.6174

### L1 — frozen 1120 inference (training-free)
- Platform experiment: `run1 probe res1120sq`, 2026-07-05, COMPLETED
- Config file: `code/configs/l1_frozen_1120.yaml`
- Command:
  ```
  python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip \
      --inference.resolution 1120 --inference.no-compile
  ```
- Input 1120 × 1120 square, PE interpolated · threshold 0.01 · num_select 300 · eager · no parameter update
- AP 0.3654 · AP50 0.4879 · AP75 0.3857 · AP_S 0.0660 · AP_M 0.1924 · AP_L 0.5034
- AR@1 0.3827 · AR@10 0.6063 · AR@100 0.6520

### L2 — rectangular input
- Platform experiment: `run1 probe res1280x736`, 2026-07-03/04, COMPLETED
- Config file: `code/configs/l2_rect_1280x736.yaml`
- Command:
  ```
  python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip \
      --inference.resolution 1280x736 --inference.no-compile
  ```
- The CLI parses rectangular resolutions in H × W order, so the actual input was
  **1280 high × 736 wide** — a tall input, not an aspect-ratio-preserving one.
- threshold 0.01 · num_select 300 · eager · no parameter update
- AP 0.3057 · AP50 0.4419 · AP75 0.3173 · AP_S 0.0501 · AP_M 0.1447 · AP_L 0.4337
- AR: not retained

### L3 — 1120px warm-start fine-tuning
- Platform experiment: `ft1120b v5 6ep autobench`, 2026-07-10, COMPLETED
- Config file: `code/configs/l3_finetune_1120.yaml`
- Command:
  ```
  python scripts/train.py --model-path ./pretrained_models/run1_best.zip \
      --epochs 6 --batch-size 1 --grad-accumulation-steps 16 \
      --learning-rate 1.5e-5 --resolution 1120 --aug-preset dg_crosscity_v2 \
      --inference-model-name checkpoint_best_total \
      --inference-config.resolution 1120 --inference-config.no-compile \
      --inference-config.threshold 0.05
  ```
- `--learning-rate` sets the **decoder** LR only; the encoder LR (1e-4), warmup
  and cosine minimum-LR factor were left at the `train.py` defaults. The exact
  warmup value in force for this run was not separately retained.
- **threshold 0.05** (the only run not at 0.01) · num_select 300 · eager · parameters updated
- In-domain source validation AP: base 0.767 → fine-tuned 0.789
- AP 0.3470 · AP50 0.4626 · AP75 0.3650 · AP_S 0.0678 · AP_M 0.1832 · AP_L 0.4811
- AR@1 0.3220 · AR@10 0.5131 · AR@100 0.5320

### L4 — gray-world white balancing at 1120
- Platform experiment: `run1 gw1120 probe`, 2026-07-09, COMPLETED
- Config file: `code/configs/l4_grayworld_1120.yaml`
- Command:
  ```
  python scripts/benchmark.py --model-path ./pretrained_models/run1_best.zip \
      --inference.resolution 1120 --inference.grayworld --inference.no-compile
  ```
- threshold 0.01 · num_select 300 · eager · no parameter update
- AP 0.3647 · AP50 0.4870 · AP75 0.3847 · AP_S 0.0658 · AP_M 0.1917 · AP_L 0.5031
- AR: not retained

### Base training run (produces the shared checkpoint)
- Platform experiment: `Large run1 80ep`, 2026-06-28, COMPLETED
- Config file: `code/configs/base_704.yaml`
- Command:
  ```
  python scripts/train.py --model-path ./pretrained_models/RFDETRLarge.zip \
      --epochs 80 --batch-size 1 --grad-accumulation-steps 16 \
      --learning-rate 7e-5 --aug-preset dg_crosscity_v2 \
      --inference-model-name checkpoint_best_total
  ```
- Resolution 704 is the model default (no `--resolution` passed). Encoder LR,
  warmup and weight decay were left at the `train.py` defaults.

## Runs that are NOT part of L0–L4

These appear in the same platform account but are not reported in the paper:

| Experiment | Date | Why excluded |
|---|---|---|
| `run1 ns600 hflipTTA` | 2026-07-03 | `--inference.num-select 600 --tta-hflip`; a separate TTA probe |
| `smoke ft1120 warmstart` | 2026-07-03 | 1-epoch, 300-sample smoke test |
| `ft1120 run1warm 15ep` | 2026-07-05 | FAILED |
| `ft1120 bench 1120 ckpt` | 2026-07-09 | FAILED |
| `decisive pseudo1120 UDA 1ep` | 2026-07-11 | pseudo-label self-training; exploratory only |
| earlier `Track6 RFDETR/DG Medium` runs | 2026-06 | superseded medium-backbone exploration |

## Code snapshot vs. current utilities

The repository contains post-challenge engineering utilities and research paths
(for example the optional `--pseudo-label` self-training branch in `train.py`,
`scripts/wbf_ensemble.py`, and everything under `extra_experiments/`) in addition
to the configurations of the reported runs. **The reported scores are tied to the
verbatim commands and configuration snapshots listed above, not to every current
utility default.** Where a training-run setting could not be confirmed from the
archived records, it is documented here as not retained rather than guessed.
