# Notice — upstream code and modifications

## Upstream

`code/` is derived from the public Hafnia **object-detection trainer package**
template (the `milestone-hafnia` trainer-package family), released under the
**MIT License**, `Copyright (c) 2025 Data-insight-Platform`. The original licence
text is retained verbatim in `LICENSE` and `code/LICENSE`, and applies to this
derivative work.

That template in turn wraps [`rfdetr`](https://github.com/roboflow/rf-detr) by
Roboflow. RF-DETR uses a split licensing model: the base package and its
Apache-designated weights are under **Apache License 2.0**, while the
`rfdetr_plus` components and the RF-DETR-XL / 2XL models are under **PML 1.0**.
No model weights of any kind are distributed in this repository, so only the
source-level Apache 2.0 terms are relevant here. Consult the upstream
[RF-DETR LICENSE](https://github.com/roboflow/rf-detr/blob/main/LICENSE) for
authoritative terms.

## Modifications made in this repository

Relative to the upstream template:

- `src/trainer_object_detection/wrapped_model.py` — added an inference-time
  resolution override (`InferenceConfig.resolution`, parsed in `H×W` order),
  gray-world channel normalization, horizontal-flip TTA, and a `num_select`
  postprocessor override; the RF-DETR optimization path is skipped whenever a
  resolution override is set, because compiled graphs are pinned to the
  training-time square resolution.
- `scripts/train.py` — exposed warmup length, encoder learning rate and the
  cosine minimum-LR factor as CLI arguments; retained an optional
  `--pseudo-label` self-training path.
- `scripts/package_predictions.py` — added as a submission-packaging CLI
  (confidence floor, per-class top-K truncation, coordinate and confidence
  rounding).
- `scripts/wbf_ensemble.py` — added as a standalone weighted-box-fusion utility.
- `configs/` — added; these YAML files record the exact settings of each reported
  configuration and are documentation, not code the scripts read.
- `extra_experiments/` — working-repository scripts kept for completeness.
- Documentation (`README.md`, `SUPPLEMENTARY.md`, `RESULTS.md`, `RUN_MANIFEST.md`,
  this file) added or rewritten. The RF-DETR citation in `code/README.md` was
  updated from the upstream pre-acceptance `@misc` arXiv form to the accepted
  ICLR 2026 venue, so it matches the citation in our paper.

## Not redistributable, therefore absent

- **Model weights.** No checkpoint, EMA weight or exported model is included.
- **Datasets.** The Hafnia managed training dataset and the hidden Track 6
  evaluation benchmark are subject to the challenge platform's access and
  redistribution conditions and cannot be republished here. Benchmark images,
  ground-truth annotations and per-image evaluation outcomes were never exposed
  to participants.

## Scope of the code

The pseudo-label / self-training path retained in `scripts/train.py` was **not**
used to produce any reported L0–L4 result; it was exercised only as a separate
exploratory experiment. `scripts/wbf_ensemble.py` and everything under
`extra_experiments/` are likewise unused for the reported results.
