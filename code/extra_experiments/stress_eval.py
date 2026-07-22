"""On-platform STRESS EVALUATION harness for AI City Track 6.

Evaluates a trained checkpoint on the SOURCE *validation* split under a battery of
GEOMETRY-PRESERVING synthetic corruptions (illumination / weather / blur / noise /
compression), reporting COCO mAP per corruption. It runs INSIDE the Hafnia
container where the dataset + ground truth live, so it consumes **NO** eval-server
submission slot (it is an ordinary experiment, sequential with the 1-concurrent limit).

WHY (honest framing): this is a NEGATIVE FILTER, not a cross-city oracle.
  * Catches augmentation / threshold / num_select choices that DESTROY robustness.
  * Ranks candidate checkpoints (run1 @704 vs run2 @768) by RELATIVE robustness — the
    one that degrades least under corruption is the better cross-city bet.
  * Doubles as the offline check for hflip / threshold / num_select effects on val.
Synthetic-corruption robustness != real cross-city domain shift (different cameras,
geography, vehicle distribution, backgrounds). Use it to REJECT disasters and to rank,
NOT to predict the hidden target-city number. There is no local cross-city GT.

Corruptions are PHOTOMETRIC/BLUR/NOISE ONLY (no spatial warps), so the val GT boxes
stay valid without remapping. They use the package-pinned albumentations 2.0.8 (signatures
+ functional apply verified), fixed/narrow severities, and a fixed Compose seed so the
SAME corruption hits the SAME image across different checkpoints (apples-to-apples).

Usage (after run1/run2 completes, attach the checkpoint on the platform OR pass --model-path):
    python scripts/stress_eval.py                          # platform-attached checkpoint
    python scripts/stress_eval.py --model-path ./pretrained_models/RFDETRLarge.zip --samples 50  # local smoke
    python scripts/stress_eval.py --samples 1500           # cap val for speed (recommended on platform)
    python scripts/stress_eval.py --inference-config.hflip true   # also probe hflip's effect on val
"""
from typing import Annotated, Callable, Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Primitive
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="stress_eval", help="Source-val stress evaluation under synthetic corruptions")

# Fixed seed: makes the corruption sampling reproducible AND identical across checkpoints
# (so run1 and run2 are stressed with the exact same corrupted images, in val iteration order).
DEFAULT_SEED = 1234


def _build_corruptions(seed: int) -> "Dict[str, Optional[Callable[[np.ndarray], np.ndarray]]]":
    """Return {name: fn(uint8 HWC)->uint8 HWC} for each corruption ('clean' maps to None).

    albumentations is imported lazily so this module imports fine in environments without it
    (e.g. the local rfdetr dev env); it is only needed at runtime inside the Hafnia container.
    Severities are fixed/narrow (geometry-preserving) — verified on albumentations 2.0.8.
    """
    import albumentations as A

    def comp(transform):
        # Compose(seed=...) is supported on albumentations 2.0.x for reproducibility;
        # fall back gracefully if a build predates it.
        try:
            return A.Compose([transform], seed=seed)
        except TypeError:
            return A.Compose([transform])

    transforms = {
        "lowlight": A.RandomBrightnessContrast(brightness_limit=(-0.55, -0.45), contrast_limit=(-0.10, 0.0), p=1.0),
        "bright": A.RandomBrightnessContrast(brightness_limit=(0.45, 0.55), contrast_limit=(0.0, 0.10), p=1.0),
        "low_contrast": A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.60, -0.50), p=1.0),
        "fog": A.RandomFog(fog_coef_range=(0.4, 0.6), alpha_coef=0.1, p=1.0),
        "rain": A.RandomRain(brightness_coefficient=0.8, drop_width=1, blur_value=3, rain_type="default", p=1.0),
        "motion_blur": A.MotionBlur(blur_limit=(11, 13), p=1.0),
        "jpeg": A.ImageCompression(quality_range=(25, 30), p=1.0),
        "iso_noise": A.ISONoise(color_shift=(0.05, 0.06), intensity=(0.4, 0.5), p=1.0),
        "gauss_noise": A.GaussNoise(std_range=(0.08, 0.10), p=1.0),
    }

    funcs: "Dict[str, Optional[Callable[[np.ndarray], np.ndarray]]]" = {"clean": None}
    for name, transform in transforms.items():
        composed = comp(transform)

        def make(c):
            def apply(img: np.ndarray) -> np.ndarray:
                return np.ascontiguousarray(c(image=img)["image"])

            return apply

        funcs[name] = make(composed)
    return funcs


class _CorruptingModel(InferenceModel):
    """Wraps a WrappedModel and applies a fixed corruption to each image before inference.

    The corruption mutates only pixel appearance (not geometry), so it is applied to the
    raw image the harness reads; the underlying model's own preprocessing/compile path and
    the dataset's GT boxes are unaffected.
    """

    def __init__(self, base: WrappedModel, corruption: "Optional[Callable[[np.ndarray], np.ndarray]]"):
        self.base = base
        self.corruption = corruption

    def get_model_info(self) -> ModelInfo:
        return self.base.get_model_info()

    def predict(self, images: Union[ImageType, List[ImageType]], sample_dict: Optional[dict] = None) -> List[Primitive]:
        if self.corruption is not None and isinstance(images, np.ndarray):
            images = self.corruption(images)
        return self.base.predict(images, sample_dict=sample_dict)


def _primary_map(metrics: Dict[str, float]) -> float:
    """Pull the COCO primary mAP@[.5:.95] from the (task-prefixed) metric dict."""
    for key, value in metrics.items():
        if key.split("/")[-1] == "mAP":
            return float(value)
    return float("nan")


def _sub(metrics: Dict[str, float], leaf: str) -> float:
    for key, value in metrics.items():
        if key.split("/")[-1] == leaf:
            return float(value)
    return float("nan")


@app.default
def main(
    project_name: Annotated[str, Parameter(help="Project name for the experiment")] = "Stress Eval RF-DETR",
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to a compressed (zip) model archive to evaluate. IGNORED when a checkpoint is "
                "attached to the experiment on the Hafnia platform (that checkpoint is used instead)."
            )
        ),
    ] = "./pretrained_models/RFDETRLarge.zip",
    samples: Annotated[
        Optional[int],
        Parameter(help="Cap the number of VAL samples (deterministic subset, seed 42). Omit to use all. ~1500 is a good platform default."),
    ] = None,
    inference_config: Annotated[
        Optional[InferenceConfig],
        Parameter(help="Inference config (threshold / num_select / hflip / compile). Reuse to probe their effect on val."),
    ] = None,
    seed: Annotated[int, Parameter(help="Corruption sampling seed (identical across checkpoints for fair comparison)")] = DEFAULT_SEED,
    corruptions: Annotated[
        Optional[str],
        Parameter(help="Comma-separated subset of corruptions to run (default: all). 'clean' is always included."),
    ] = None,
):
    """Evaluate a checkpoint on corrupted SOURCE-VAL and log per-corruption COCO mAP.

    Loads the dataset (hidden dataset on the platform, small public sample locally), takes the
    VAL split (which has ground truth), and for each corruption runs inference + COCO metrics.
    Emits per-corruption mAP/mAP_m/mAP_l/AR_100 plus a clean-vs-corruption robustness summary.
    """
    inference_config = inference_config or InferenceConfig()
    has_cuda = torch.cuda.is_available()
    user_logger.info(f"CUDA available: {has_cuda}")

    logger = HafniaLogger(project_name=project_name)

    if hafnia_utils.is_hafnia_cloud_job():
        dataset = HafniaDataset.from_path(hafnia_utils.get_dataset_path_in_hafnia_cloud())
    else:
        dataset = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")

    val = dataset.create_split_dataset(split_name=SplitName.VAL)
    if samples is not None:
        val = val.select_samples(n_samples=samples)  # deterministic (seed 42) -> same subset across checkpoints
    user_logger.info(f"VAL samples for stress eval: {len(val)}")

    # Resolve the checkpoint: platform-attached checkpoint wins over --model-path (mirrors train.py).
    checkpoint = utils.get_checkpoint_if_available(logger)
    zip_path = checkpoint.as_posix() if checkpoint is not None else model_path
    user_logger.info(f"Evaluating checkpoint archive: {zip_path}")

    model = WrappedModel.load_model(zip_path, inference_config=inference_config)
    model.optimize_for_inference()

    # GT presence guard: VAL must carry ground-truth boxes for COCO metrics.
    gt_col = model.task.primitive.column_name()
    no_gt = val.samples.select(pl.col(gt_col).list.len()).sum().item() == 0
    if no_gt:
        user_logger.warning(
            f"No ground-truth '{gt_col}' found in the VAL split — cannot compute stress metrics. Aborting."
        )
        return logger

    all_corruptions = _build_corruptions(seed)
    if corruptions is not None:
        wanted = {c.strip() for c in corruptions.split(",") if c.strip()}
        unknown = wanted - set(all_corruptions)
        if unknown:
            raise ValueError(f"Unknown corruption(s) {sorted(unknown)}. Options: {sorted(all_corruptions)}")
        run_set = ["clean"] + [c for c in all_corruptions if c != "clean" and c in wanted]
    else:
        run_set = list(all_corruptions)

    user_logger.info(
        f"Inference config: threshold={inference_config.threshold} num_select={inference_config.num_select} "
        f"hflip={inference_config.hflip} compile={inference_config.compile}"
    )
    user_logger.info(f"Running {len(run_set)} passes: {run_set}")

    results: Dict[str, Dict[str, float]] = {}
    for cname in run_set:
        corrupt_model = _CorruptingModel(model, all_corruptions[cname])
        predicted = run_inference_on_dataset(dataset=val, model=corrupt_model)
        metrics = metric_calculations(prediction_dataset=predicted)
        row = {
            "mAP": _primary_map(metrics),
            "mAP_m": _sub(metrics, "mAP_m"),
            "mAP_l": _sub(metrics, "mAP_l"),
            "AR_100": _sub(metrics, "AR_100"),
        }
        results[cname] = row
        for leaf, val_ in row.items():
            utils.safe_log_metric(logger, f"stress/{cname}/{leaf}", val_)
        user_logger.info(
            f"  [{cname:12s}] mAP={row['mAP']:.4f}  mAP_m={row['mAP_m']:.4f}  "
            f"mAP_l={row['mAP_l']:.4f}  AR100={row['AR_100']:.4f}"
        )

    # ---- Robustness summary (clean vs corruptions) ----
    clean_map = results.get("clean", {}).get("mAP", float("nan"))
    corr = {k: v["mAP"] for k, v in results.items() if k != "clean"}
    mean_corr = float(np.mean(list(corr.values()))) if corr else float("nan")
    worst_name = min(corr, key=corr.get) if corr else None
    worst_map = corr[worst_name] if worst_name else float("nan")
    robustness_ratio = (mean_corr / clean_map) if (clean_map and clean_map == clean_map and clean_map > 0) else float("nan")

    utils.safe_log_metric(logger, "stress_summary/clean_mAP", clean_map)
    utils.safe_log_metric(logger, "stress_summary/mean_corruption_mAP", mean_corr)
    utils.safe_log_metric(logger, "stress_summary/worst_corruption_mAP", worst_map)
    utils.safe_log_metric(logger, "stress_summary/robustness_ratio", robustness_ratio)

    print("\n=== STRESS EVAL SUMMARY ===", flush=True)
    print(f"checkpoint: {zip_path}", flush=True)
    print(
        f"inference: threshold={inference_config.threshold} num_select={inference_config.num_select} "
        f"hflip={inference_config.hflip}",
        flush=True,
    )
    print(f"{'corruption':14s} {'mAP':>8s} {'mAP_m':>8s} {'mAP_l':>8s} {'AR_100':>8s} {'drop_vs_clean':>14s}", flush=True)
    for cname in run_set:
        r = results[cname]
        drop = (clean_map - r["mAP"]) if (cname != "clean" and clean_map == clean_map) else 0.0
        print(
            f"{cname:14s} {r['mAP']:8.4f} {r['mAP_m']:8.4f} {r['mAP_l']:8.4f} {r['AR_100']:8.4f} {drop:14.4f}",
            flush=True,
        )
    print(f"\nclean mAP            : {clean_map:.4f}", flush=True)
    print(f"mean corruption mAP  : {mean_corr:.4f}", flush=True)
    print(f"worst corruption     : {worst_name} ({worst_map:.4f})", flush=True)
    print(f"robustness ratio     : {robustness_ratio:.4f}  (mean_corruption / clean; higher = more robust)", flush=True)
    print("NOTE: negative filter only — synthetic robustness != real cross-city shift.", flush=True)

    return logger


if __name__ == "__main__":
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL, order=2)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")
    app()
