# aicity2026-track6

Code, configuration files and inference-packaging scripts for:

> **Frozen High-Resolution Inference for Cross-City Object Detection: An AI City
> Challenge 2026 Study**
> AI City Challenge 2026 Workshop @ ECCV 2026

A single RF-DETR-Large detector is evaluated under five configurations on the
AI City Challenge 2026 Track 6 hidden benchmark. The headline result is that
**frozen 1120 × 1120 inference of a 704-trained checkpoint** — zero parameter
updates — reached the highest aggregate AP among the configurations we tested
(0.3272 → 0.3654), above the one high-resolution fine-tuning recipe we ran.

| | |
|---|---|
| **Results** | [`RESULTS.md`](RESULTS.md) — every configuration and score |
| **Run provenance** | [`RUN_MANIFEST.md`](RUN_MANIFEST.md) — verbatim platform command, settings and scores for every reported run |
| **Full settings** | [`SUPPLEMENTARY.md`](SUPPLEMENTARY.md) — training, inference and packaging configuration |
| **Upstream & modifications** | [`NOTICE.md`](NOTICE.md) |
| **Licence** | MIT (see [`LICENSE`](LICENSE) and `NOTICE.md`) |

## Layout

```
RESULTS.md          reported scores for L0-L4
SUPPLEMENTARY.md    complete configuration documentation
NOTICE.md           upstream provenance + list of modifications
code/
  configs/          per-configuration YAML (documentation of exact settings)
  scripts/          train.py, benchmark.py, package_predictions.py, ...
  src/              trainer_object_detection wrapper
  extra_experiments/  working-repo scripts, unused for reported results
  Dockerfile, pyproject.toml, uv.lock
```

## What is not here

No model weights and no datasets. The Hafnia managed training dataset and the
hidden Track 6 evaluation benchmark are subject to the challenge platform's
access and redistribution conditions; benchmark images, ground-truth annotations
and per-image evaluation outcomes were never exposed to participants. Reproducing
the reported numbers therefore requires access to the Hafnia platform and the
allowed pretrained RF-DETR weights. See `NOTICE.md`.

## Caveats worth reading before citing

All scores are **aggregate-only** over a hidden source/target mixture — no
per-domain metric exists. Each configuration was submitted **once**, with no fixed
random seed, so there are no error bars. L1 and L3 are **not** a matched control:
besides the optimization differences, L3 was scored at confidence threshold 0.05
while L0/L1/L2/L4 used 0.01, and L0 ran on the compiled path while L1–L4 ran
eagerly. `RUN_MANIFEST.md` records both.
`SUPPLEMENTARY.md` and the paper's Limitations section state these in full.
