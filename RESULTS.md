# Results

All scores are the aggregate COCO-style AP returned by the AI City Challenge 2026
Track 6 evaluation server over a **hidden mixture of source-city and target-city
images**. The server does not separate the two domains, so no per-domain number
exists. Each configuration was submitted **once** (`n = 1`); no fixed random seed
was set and no error bars are available.

## Configurations

| ID | Checkpoint | Input | Parameter update | AP |
|---|---|---|---|---:|
| L0 | shared 704 ckpt | 704 × 704 | No | 0.3272 |
| **L1** | shared 704 ckpt | **1120 × 1120** | No | **0.3654** |
| L2 | shared 704 ckpt | 1280(H) × 736(W), rectangular | No | 0.3057 |
| L3 | shared 704 ckpt, warm-start | 1120 × 1120, fine-tuned | **Yes** | 0.3470 |
| L4 | shared 704 ckpt | 1120 × 1120 + gray-world | No | 0.3647 |

L0, L1, L2 and L4 use the same 704-trained checkpoint (`checkpoint_best_total`,
the higher-scoring of the EMA and regular weights) and perform no parameter
updates. Their input processing and execution modes are listed in
`RUN_MANIFEST.md`. L3 warm-starts from that checkpoint and updates the model
parameters.

## Precision breakdown

| ID | AP | ΔAP vs L0 | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.3272 | — | 0.4391 | 0.3335 | 0.0519 | 0.1528 | 0.4687 |
| **L1** | **0.3654** | **+0.0382** | **0.4879** | **0.3857** | 0.0660 | **0.1924** | **0.5034** |
| L4 | 0.3647 | +0.0375 | 0.4870 | 0.3847 | 0.0658 | 0.1917 | 0.5031 |
| L3 | 0.3470 | +0.0198 | 0.4626 | 0.3650 | **0.0678** | 0.1832 | 0.4811 |
| L2 | 0.3057 | −0.0215 | 0.4419 | 0.3173 | 0.0501 | 0.1447 | 0.4337 |

## Recall

AR was retained only for L0, L1 and L3 in the archived server reports. The
evaluation server is now closed, so the missing L2/L4 AR values cannot be
recovered or recomputed.

| ID | AR@1 | AR@10 | AR@100 |
|---|---:|---:|---:|
| L0 | 0.3568 | 0.5782 | 0.6174 |
| L1 | 0.3827 | 0.6063 | 0.6520 |
| L3 | 0.3220 | 0.5131 | 0.5320 |

## Reading these numbers

- L1 (frozen 1120 inference, **zero** parameter updates) is the highest tested
  aggregate AP, +0.0382 over the 704 base.
- The gain is largest **relatively** on small objects (AP<sub>S</sub> +27.2 %) and
  largest **absolutely** on medium objects (AP<sub>M</sub> +0.0396).
- L3 (the one fine-tuning recipe evaluated) raised in-domain validation AP
  (0.767 → 0.789) but reached a lower aggregate benchmark AP than L1, together
  with a substantial recall drop (AR@100 0.6520 → 0.5320).
- L2's configuration string `1280x736` is parsed by the benchmark CLI in **H × W**
  order, so the actual model input was 1280 high × 736 wide. Orientation, aspect
  ratio, image shape and pixel count all differ from the square 1120 setting at
  once — this is **not** an aspect-ratio-preserving or native-aspect input.

## What these numbers do not show

- No target-city-specific improvement is established; every statement is
  benchmark-level, because the server metric is aggregate-only.
- L1 vs L3 is **not** a matched control: they differ in optimizer updates, training
  duration, scheduler state, checkpoint selection **and the evaluation confidence
  threshold**. L0/L1/L2/L4 were scored at the CLI default `0.01`; the L3 run's
  built-in evaluation pass was launched with `--inference-config.threshold 0.05`.
  Because COCO AP is rank-based under a per-image detection cap, the higher
  threshold is expected to work *against* L3. The magnitude is unmeasured — the
  evaluation server is closed, so L3 cannot be re-scored at `0.01`. Verbatim
  platform commands for all five runs are in `SUPPLEMENTARY.md`.
- Single-model, single-benchmark, single-run results. Do not generalize to other
  detectors or domains.

See `SUPPLEMENTARY.md` for every configuration setting and the paper for the full
discussion and limitations.
